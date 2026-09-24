"""ACP TraceSink：把一轮 ReAct 过程转成 ``session/update``。"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from llgraph.console.runtime.sse_sink import _step_to_dict
from llgraph.display.trace_sink import strip_ansi
from llgraph.editor.acp.updates import (
    agent_message_chunk,
    agent_thought_chunk,
    is_tool_call_step,
    tool_call_diffs,
    tool_call_from_step,
    tool_call_locations,
    tool_call_pending,
    tool_call_status,
)

# 一次调用最多留几份文件的改前 / 改后全文：写工具一次只改一份，
# 留这个上限是为了万一有工具循环写，不至于把一轮的内存堆满
_MAX_EDITS_PER_CALL = 8


class AcpTraceSink:
    """
    TraceSink 实现：正文走 ``agent_message_chunk``，思考走 ``agent_thought_chunk``，
    工具走 ``tool_call`` 三段（pending → in_progress → completed / failed）。

    三段各自的来源不同：pending 来自模型决策（``tool_calls_planned``），
    in_progress 来自 ToolNode 真正开跑（``tool_started``，在工具线程里被调用），
    completed 来自跑完登记的 trace 步骤（``step_added``）。
    三者靠模型给的 tool_call id 串成一条，编辑器里始终只有一行在动。
    写工具落地时另报一次改动（``tool_edited``），收尾那条据此带上 diff 块。

    trace 行（``line()``）不外发：ACP 没有对应的更新类型，编辑器里刷成思考会很吵。
    """

    preserves_ansi: bool = False
    suppress_reply_body: bool = True
    suppress_web_hints: bool = True

    def __init__(
        self,
        send_update: Callable[[dict[str, Any]], None],
        *,
        id_prefix: str = "",
        workspace: Path | None = None,
    ) -> None:
        """
        @param send_update ``session/update`` 的 update 字段回调
        @param id_prefix toolCallId 前缀；同一会话里每轮换一个，免得第二轮的
            ``call_1`` 撞上第一轮那条已经收尾的调用
        @param workspace 工作区根；把工具参数里的相对路径补成 ACP 要的绝对路径
        """
        self._send = send_update
        self._id_prefix = id_prefix
        self._workspace = Path(workspace) if workspace is not None else None
        self.log_lines: list[str] = []
        self.streamed_chars: int = 0
        self._thinking_sent: int = 0
        self._tool_seq: int = 0
        # 工具在 LangGraph 线程池里跑，起步通知与主线程的登记并发
        self._tool_lock = threading.Lock()
        self._tool_ids: dict[str, str] = {}
        self._open_ids: set[str] = set()
        # 写工具落地时报上来的改动，按模型给的 tool_call_id 攒着，等这次调用收尾时发出
        self._edits: dict[str, list[dict[str, Any]]] = {}

    def line(self, text: str) -> None:
        """@param text trace 行（仅留档，不外发）"""
        if text:
            self.log_lines.append(strip_ansi(text))

    def stream(self, text: str) -> None:
        """@param text 助手正文流式片段"""
        if not text:
            return
        self.streamed_chars += len(text)
        self._send(agent_message_chunk(text))

    def stream_end(self) -> None:
        """流式段落结束（ACP 无对应事件）。"""

    def thinking_update(self, text: str) -> None:
        """
        思考全文刷新：只把新增后缀发出去。

        @param text 当前思考全文
        """
        if not text:
            return
        plain = strip_ansi(text)
        if len(plain) <= self._thinking_sent:
            return
        delta = plain[self._thinking_sent :]
        self._thinking_sent = len(plain)
        if delta.strip():
            self._send(agent_thought_chunk(delta))

    def _next_tool_call_id(self) -> str:
        self._tool_seq += 1
        return f"{self._id_prefix}call_{self._tool_seq}"

    def tool_calls_planned(self, calls: list[dict[str, Any]]) -> None:
        """
        模型决定要调的工具：先各画一条 pending。

        @param calls ``[{"id": ..., "name": ..., "title": ..., "paths": [...]}]``
        """
        payloads: list[dict[str, Any]] = []
        with self._tool_lock:
            for call in calls:
                raw_id = str(call.get("id") or "").strip()
                if not raw_id or raw_id in self._tool_ids:
                    continue
                acp_id = self._next_tool_call_id()
                self._tool_ids[raw_id] = acp_id
                self._open_ids.add(acp_id)
                payloads.append(
                    tool_call_pending(
                        acp_id,
                        title=str(call.get("title") or "").strip(),
                        tool_name=str(call.get("name") or "").strip(),
                        locations=tool_call_locations(
                            call.get("paths"), self._workspace
                        ),
                    )
                )
        for payload in payloads:
            self._send(payload)

    def tool_started(self, tool_call_id: str, tool_name: str) -> None:
        """
        某次工具调用真正开跑（在工具线程里被调用）。

        没报过 pending 的 id 不发 in_progress：那说明这条调用不会有人收尾
        （trace 静默、或 explore 那类另走 explore 步骤），留一条转圈的记录更糟。

        @param tool_call_id 模型给的 id
        @param tool_name 工具名（此处不用，仅对齐观察者签名）
        """
        _ = tool_name
        with self._tool_lock:
            acp_id = self._tool_ids.get(str(tool_call_id or "").strip())
            if acp_id is None or acp_id not in self._open_ids:
                return
        self._send(tool_call_status(acp_id, "in_progress"))

    def tool_edited(self, edit: Any) -> None:
        """
        某次工具调用改了一份文件（在工具线程里被调用，写已经落地了）。

        改动此刻攒着不发：这次调用还没收尾，单发一条 ``tool_call_update``
        会在编辑器里多出一次刷新，而收尾那条本来就要带 content。

        同一份文件被同一次调用改两回时合成一条（最早的改前 + 最后的改后），
        编辑器里一次调用挂两个同名 diff 只会让人以为改了两个文件。

        @param edit ``ToolEdit``（tool_call_id / path / old_text / new_text）
        """
        raw_id = str(getattr(edit, "tool_call_id", "") or "").strip()
        path = str(getattr(edit, "path", "") or "").strip()
        if not raw_id or not path:
            return
        with self._tool_lock:
            edits = self._edits.setdefault(raw_id, [])
            for existing in edits:
                if existing["path"] == path:
                    existing["new_text"] = getattr(edit, "new_text", "")
                    return
            if len(edits) >= _MAX_EDITS_PER_CALL:
                return
            edits.append(
                {
                    "path": path,
                    "old_text": getattr(edit, "old_text", ""),
                    "new_text": getattr(edit, "new_text", ""),
                }
            )

    def step_added(self, step: Any) -> None:
        """@param step TraceStepRecord"""
        payload = _step_to_dict(step)
        if not is_tool_call_step(payload):
            return
        raw_id = str(payload.get("tool_call_id") or "").strip()
        with self._tool_lock:
            acp_id = self._tool_ids.get(raw_id) if raw_id else None
            as_update = acp_id is not None
            if acp_id is None:
                acp_id = self._next_tool_call_id()
                if raw_id:
                    self._tool_ids[raw_id] = acp_id
            self._open_ids.discard(acp_id)
            edits = self._edits.pop(raw_id, []) if raw_id else []
        update = tool_call_from_step(
            payload,
            tool_call_id=acp_id,
            as_update=as_update,
            diffs=tool_call_diffs(edits, self._workspace),
        )
        if update is not None:
            self._send(update)

    def abandon_open_tool_calls(self) -> None:
        """
        一轮收场时把还没收尾的调用标成 failed。

        被取消、或模型那轮中途出错时，编辑器里那几条 pending / in_progress
        会一直转圈——本轮之后再也不会有人来改它们的状态。
        """
        with self._tool_lock:
            open_ids = sorted(self._open_ids)
            self._open_ids.clear()
        for acp_id in open_ids:
            try:
                self._send(tool_call_status(acp_id, "failed"))
            except Exception:  # noqa: BLE001 - 连接可能已经断了，收场不能再抛
                return

    def step_selected(self, step_id: int) -> None:
        """@param step_id 步骤编号（ACP 不使用）"""
