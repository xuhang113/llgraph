"""ACP 授权弹窗：写工具 / shell 落地前问编辑器里的人一句。

ACP 的 ``session/request_permission`` 是 Agent → Client 方向的请求，
所以这是本实现里第一条反向链路：工作线程发出并阻塞等待，读循环收回包唤醒它
（等待期间 ``session/cancel`` 仍能进来，因为读循环没被占住）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from llgraph.editor.acp.jsonrpc import (
    JsonRpcConnection,
    RequestCancelled,
    RequestFailed,
)
from llgraph.permissions.approval import (
    APPROVAL_KIND_EDIT,
    ApprovalDecision,
    ApprovalRequest,
)

PERMISSION_OPTIONS: list[dict[str, str]] = [
    {"optionId": "allow_once", "name": "允许这一次", "kind": "allow_once"},
    {"optionId": "allow_always", "name": "本会话都允许", "kind": "allow_always"},
    {"optionId": "reject_once", "name": "拒绝", "kind": "reject_once"},
    {"optionId": "reject_always", "name": "本会话都拒绝", "kind": "reject_always"},
]

_DEFAULT_TIMEOUT_SEC = 600.0


class AcpPermissionGate:
    """
    一个 ACP 会话的授权闸门。

    「本会话都允许 / 都拒绝」记在闸门上，所以它按会话建、不按轮次建：
    用户点过一次「都允许」，后面几轮就不该再被问。
    """

    def __init__(
        self,
        connection: JsonRpcConnection,
        session_id: str,
        *,
        workspace: Path,
        cancel_check: Callable[[], bool] | None = None,
        timeout_sec: float | None = _DEFAULT_TIMEOUT_SEC,
    ) -> None:
        self._conn = connection
        self._session_id = session_id
        self._workspace = Path(workspace)
        self._cancel_check = cancel_check
        self._timeout_sec = timeout_sec
        self._remembered: dict[str, bool] = {}
        self._seq = 0

    # ---- 闸门 ----

    def ask(self, req: ApprovalRequest) -> ApprovalDecision:
        """
        问编辑器要不要放行这次动作（会阻塞到用户点按钮）。

        @param req 动作
        @return 决定
        """
        if self._cancelled():
            return ApprovalDecision(allowed=False, cancelled=True)
        remembered = self._remembered.get(req.kind)
        if remembered is not None:
            return ApprovalDecision(
                allowed=remembered,
                reason="" if remembered else "本会话此前选择了「都拒绝」",
            )
        try:
            result = self._conn.request(
                "session/request_permission",
                {
                    "sessionId": self._session_id,
                    "toolCall": self._tool_call(req),
                    "options": PERMISSION_OPTIONS,
                },
                timeout=self._timeout_sec,
                cancel_check=self._cancel_check,
            )
        except RequestCancelled:
            return ApprovalDecision(allowed=False, cancelled=True)
        except RequestFailed as exc:
            return ApprovalDecision(
                allowed=False, reason=f"编辑器没能处理授权请求（{exc}）"
            )
        return self._decide(req, result)

    # ---- 载荷与回包 ----

    def _cancelled(self) -> bool:
        return self._cancel_check is not None and self._cancel_check()

    def _abs_path(self, rel: str) -> str:
        candidate = Path(rel)
        if candidate.is_absolute():
            return str(candidate)
        return str(self._workspace / candidate)

    def _tool_call(self, req: ApprovalRequest) -> dict[str, Any]:
        """
        动作 → ACP ToolCall（编辑器据此渲染弹窗内容）。

        编辑改动带 ``diff`` 块：用户要看的是「这一刀改了什么」，
        光有文件名不够，而 diff 是 ACP 里编辑器唯一会渲染成改动预览的块。

        @param req 动作
        @return toolCall 载荷
        """
        self._seq += 1
        payload: dict[str, Any] = {
            "toolCallId": f"perm_{self._seq}",
            "title": req.title(),
            "kind": req.kind,
            "status": "pending",
        }
        if req.path:
            abs_path = self._abs_path(req.path)
            payload["locations"] = [{"path": abs_path}]
            if req.kind == APPROVAL_KIND_EDIT and req.new_text is not None:
                payload["content"] = [
                    {
                        "type": "diff",
                        "path": abs_path,
                        "oldText": req.old_text if req.old_text else None,
                        "newText": req.new_text,
                    }
                ]
        elif req.command:
            payload["content"] = [
                {
                    "type": "content",
                    "content": {"type": "text", "text": req.command},
                }
            ]
        return payload

    def _decide(self, req: ApprovalRequest, result: Any) -> ApprovalDecision:
        """
        授权回包 → 决定；认不出的回包按拒绝算。

        @param req 动作（用于记住「都允许 / 都拒绝」）
        @param result ``session/request_permission`` 的 result
        @return 决定
        """
        outcome = result.get("outcome") if isinstance(result, dict) else None
        if isinstance(outcome, str):
            name, option_id = outcome, None
        elif isinstance(outcome, dict):
            name = outcome.get("outcome")
            option_id = outcome.get("optionId")
        else:
            name, option_id = None, None

        if name == "cancelled":
            return ApprovalDecision(allowed=False, cancelled=True)
        if name != "selected" or not isinstance(option_id, str) or not option_id.strip():
            return ApprovalDecision(allowed=False, reason="授权回复无法识别")

        choice = option_id.strip()
        allowed = choice.startswith("allow")
        if not allowed and not choice.startswith("reject"):
            return ApprovalDecision(allowed=False, reason=f"未知授权选项: {choice}")
        if choice.endswith("always"):
            self._remembered[req.kind] = allowed
        return ApprovalDecision(
            allowed=allowed,
            reason="" if allowed else "用户在编辑器里选择了拒绝",
        )
