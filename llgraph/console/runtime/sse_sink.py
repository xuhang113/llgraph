"""SSE TraceSink：将 trace 转为 Web 事件。"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Callable

from llgraph.display.trace_sink import strip_ansi


def _step_to_dict(step: Any) -> dict[str, Any]:
    if is_dataclass(step):
        data = asdict(step)
        usage = data.get("usage")
        if usage is not None and is_dataclass(usage):
            data["usage"] = asdict(usage)
        return data
    return {"raw": str(step)}


class SseTraceSink:
    """TraceSink 实现，回调 emit(event_dict)。"""

    preserves_ansi: bool = False
    # Web 主区域单独展示助手正文；trace 仅保留步骤/工具/完成标记
    suppress_reply_body: bool = True
    suppress_web_hints: bool = True

    def __init__(self, emit: Callable[[dict[str, Any]], None]) -> None:
        self._emit = emit
        self.log_lines: list[str] = []

    def line(self, text: str) -> None:
        """追加一行 trace。"""
        if not text:
            return
        plain = strip_ansi(text)
        self.log_lines.append(plain)
        self._emit({"type": "trace_line", "text": plain})

    def stream(self, text: str) -> None:
        """流式 assistant 片段。"""
        if text:
            self._emit({"type": "stream_delta", "text": text})

    def stream_end(self) -> None:
        """流式段落结束。"""
        self._emit({"type": "stream_end"})

    def thinking_update(self, text: str) -> None:
        """流式 thinking 全文（Web 右侧面板与对话区 live 展示）。"""
        if not text:
            return
        plain = strip_ansi(text)
        self._emit({"type": "thinking_delta", "text": plain})

    def step_added(self, step: Any) -> None:
        """新 trace 步骤。"""
        self._emit({"type": "trace_step", "step": _step_to_dict(step)})

    def step_selected(self, step_id: int) -> None:
        """步骤选中（Web 暂不使用）。"""
        self._emit({"type": "trace_step_selected", "step_id": step_id})
