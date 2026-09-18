"""ACP TraceSink：把一轮 ReAct 过程转成 ``session/update``。"""

from __future__ import annotations

from typing import Any, Callable

from llgraph.console.runtime.sse_sink import _step_to_dict
from llgraph.display.trace_sink import strip_ansi
from llgraph.editor.acp.updates import (
    agent_message_chunk,
    agent_thought_chunk,
    tool_call_from_step,
)


class AcpTraceSink:
    """
    TraceSink 实现：正文走 ``agent_message_chunk``，思考走 ``agent_thought_chunk``，
    工具步骤走 ``tool_call``。


    trace 行（``line()``）不外发：ACP 没有对应的更新类型，编辑器里刷成思考会很吵。
    """

    preserves_ansi: bool = False
    suppress_reply_body: bool = True
    suppress_web_hints: bool = True

    def __init__(self, send_update: Callable[[dict[str, Any]], None]) -> None:
        self._send = send_update
        self.log_lines: list[str] = []
        self.streamed_chars: int = 0
        self._thinking_sent: int = 0
        self._tool_seq: int = 0

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

    def step_added(self, step: Any) -> None:
        """@param step TraceStepRecord"""
        payload = _step_to_dict(step)
        self._tool_seq += 1
        update = tool_call_from_step(
            payload,
            tool_call_id=f"call_{self._tool_seq}",
        )
        if update is not None:
            self._send(update)

    def step_selected(self, step_id: int) -> None:
        """@param step_id 步骤编号（ACP 不使用）"""
