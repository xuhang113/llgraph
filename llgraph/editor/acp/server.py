"""ACP 服务端：把 llgraph Agent 接到编辑器（Zed / Neovim 等 ACP 客户端）。"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TextIO

from llgraph.editor.acp.jsonrpc import (
    DEFERRED,
    INTERNAL_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    JsonRpcConnection,
    JsonRpcError,
    invalid_params,
)
from llgraph.editor.acp.turn import AcpTurnRequest, AcpTurnResult
from llgraph.editor.acp.updates import ACP_PROTOCOL_VERSION, prompt_text

TurnRunner = Callable[..., AcpTurnResult]


@dataclass
class AcpSession:
    """一个 ACP 会话（sessionId 即 llgraph thread_id）。"""

    session_id: str
    workspace: Path
    allow_write: bool = False
    busy: bool = False
    cancelled: threading.Event = field(default_factory=threading.Event)
    permission: Any = None
    """授权闸门（``AcpPermissionGate``）；None 表示写入不逐次确认。"""


class AcpServer:
    """
    ACP 方法分发。


    ``session/prompt`` 交给工作线程，读循环继续收 ``session/cancel``——
    否则编辑器里的「停止」按钮要等一整轮跑完才生效。
    """

    def __init__(
        self,
        connection: JsonRpcConnection,
        *,
        allow_write: bool = False,
        ask_permission: bool = False,
        default_workspace: Path | None = None,
        turn_runner: TurnRunner | None = None,
    ) -> None:
        self._conn = connection
        self._allow_write = allow_write
        self._ask_permission = ask_permission and allow_write
        self._default_workspace = default_workspace
        self._turn_runner = turn_runner
        self._sessions: dict[str, AcpSession] = {}
        self._sessions_lock = threading.Lock()
        self.initialized = False

    # ---- 分发 ----

    def dispatch(self, method: str, params: dict[str, Any], request_id: Any) -> Any:
        """
        处理一条请求 / 通知。

        @param method 方法名
        @param params 参数
        @param request_id 请求 id（通知为 None）
        @return 结果、DEFERRED 或 None
        """
        if method == "initialize":
            return self._initialize(params)
        if method == "authenticate":
            return {}
        if method == "session/new":
            return self._session_new(params)
        if method == "session/prompt":
            return self._session_prompt(params, request_id)
        if method == "session/cancel":
            return self._session_cancel(params)
        raise JsonRpcError(METHOD_NOT_FOUND, f"未实现的方法: {method}")

    def serve(self) -> None:
        """跑读循环直到对端关闭连接。"""
        self._conn.serve(self.dispatch)

    # ---- 方法实现 ----

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        version = params.get("protocolVersion", ACP_PROTOCOL_VERSION)
        if not isinstance(version, int):
            raise invalid_params("protocolVersion 必须是整数")
        self.initialized = True
        return {
            # 协商取双方较小值：客户端更新时不至于被我们顶到不认识的版本
            "protocolVersion": min(version, ACP_PROTOCOL_VERSION),
            "agentCapabilities": {
                "loadSession": False,
                "promptCapabilities": {
                    "image": False,
                    "audio": False,
                    "embeddedContext": True,
                },
            },
            "authMethods": [],
        }

    def _session_new(self, params: dict[str, Any]) -> dict[str, Any]:
        cwd = params.get("cwd")
        if cwd is None and self._default_workspace is not None:
            workspace = self._default_workspace
        else:
            if not isinstance(cwd, str) or not cwd.strip():
                raise invalid_params("session/new 需要 cwd")
            candidate = Path(cwd).expanduser()
            if not candidate.is_absolute():
                raise invalid_params("cwd 必须是绝对路径")
            workspace = candidate.resolve()
        if not workspace.is_dir():
            raise invalid_params(f"工作区不是有效目录: {workspace}")

        from llgraph.console.runtime.agent_service import create_agent_session

        session_id = create_agent_session(workspace)
        session = AcpSession(
            session_id=session_id,
            workspace=workspace,
            allow_write=self._allow_write,
        )
        if self._ask_permission:
            from llgraph.editor.acp.permission import AcpPermissionGate

            # 闸门按会话建：「本会话都允许」要能跨轮记住
            session.permission = AcpPermissionGate(
                self._conn,
                session_id,
                workspace=workspace,
                cancel_check=session.cancelled.is_set,
            )
        with self._sessions_lock:
            self._sessions[session_id] = session
        return {"sessionId": session_id}

    def _require_session(self, params: dict[str, Any]) -> AcpSession:
        session_id = params.get("sessionId")
        if not isinstance(session_id, str) or not session_id.strip():
            raise invalid_params("缺少 sessionId")
        with self._sessions_lock:
            session = self._sessions.get(session_id.strip())
        if session is None:
            raise invalid_params(f"未知 sessionId: {session_id}")
        return session

    def _session_prompt(self, params: dict[str, Any], request_id: Any) -> Any:
        session = self._require_session(params)
        if request_id is None:
            raise JsonRpcError(INVALID_REQUEST, "session/prompt 必须是请求，不能是通知")
        if session.busy:
            raise JsonRpcError(INVALID_REQUEST, "该会话已有对话在进行，请先 session/cancel")
        text = prompt_text(params.get("prompt"))
        if not text:
            raise invalid_params("prompt 为空")
        session.busy = True
        session.cancelled.clear()
        worker = threading.Thread(
            target=self._run_turn,
            args=(session, text, request_id),
            daemon=True,
            name=f"acp-turn-{session.session_id[:12]}",
        )
        worker.start()
        return DEFERRED

    def _session_cancel(self, params: dict[str, Any]) -> None:
        session_id = params.get("sessionId")
        if not isinstance(session_id, str):
            return None
        with self._sessions_lock:
            session = self._sessions.get(session_id.strip())
        if session is None:
            return None
        session.cancelled.set()
        from llgraph.editor.acp.turn import request_turn_cancel

        # 光置位只能在 ReAct 步间生效；正卡在模型返回上的那一轮要另走登记表
        request_turn_cancel(session.session_id)
        return None

    # ---- 一轮对话 ----

    def _send_update(self, session: AcpSession, update: dict[str, Any]) -> None:
        self._conn.notify(
            "session/update",
            {"sessionId": session.session_id, "update": update},
        )

    def _run_turn(self, session: AcpSession, text: str, request_id: Any) -> None:
        runner = self._turn_runner
        if runner is None:
            from llgraph.editor.acp.turn import run_acp_turn

            runner = run_acp_turn
        try:
            result = runner(
                AcpTurnRequest(
                    workspace=session.workspace,
                    thread_id=session.session_id,
                    message=text,
                    allow_write=session.allow_write,
                    permission_ask=(
                        session.permission.ask
                        if session.permission is not None
                        else None
                    ),
                ),
                send_update=lambda update: self._send_update(session, update),
                cancel_check=session.cancelled.is_set,
            )
        except Exception as exc:
            self._conn.respond_error(request_id, JsonRpcError(INTERNAL_ERROR, str(exc)))
            return
        finally:
            session.busy = False
            session.cancelled.clear()
        stop_reason = result.stop_reason or "end_turn"
        self._conn.respond(request_id, {"stopReason": stop_reason})


def serve_stdio(
    *,
    allow_write: bool = False,
    ask_permission: bool = False,
    default_workspace: Path | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> None:
    """
    在 stdin/stdout 上跑 ACP。

    协议独占真正的 stdout：进程内 ``sys.stdout`` 会被改指 stderr，
    否则 Agent 链路里任何一行 print 都会插进 ndjson 流，把编辑器的解析打断。

    @param allow_write 是否允许写工作区文件
    @param ask_permission 每次写 / 执行前走 ``session/request_permission``
    @param default_workspace ``session/new`` 未给 cwd 时的兜底工作区
    @param stdin 读端（默认 sys.stdin）
    @param stdout 写端（默认真实 sys.stdout）
    """
    reader = stdin if stdin is not None else sys.stdin
    writer = stdout if stdout is not None else sys.stdout
    if stdout is None:
        sys.stdout = sys.stderr
    try:
        server = AcpServer(
            JsonRpcConnection(reader, writer),
            allow_write=allow_write,
            ask_permission=ask_permission,
            default_workspace=default_workspace,
        )
        server.serve()
    finally:
        if stdout is None:
            sys.stdout = writer
