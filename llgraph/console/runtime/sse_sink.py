"""SSE TraceSink：将 trace 转为 Web 事件。"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
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


class PersistingSseTraceSink:
    """
    包装 SseTraceSink：增量落盘 live_web_trace.json，切换会话后可恢复。

    代理 TraceSink 接口供 TraceSession 使用。
    """

    preserves_ansi: bool = False
    suppress_reply_body: bool = True
    suppress_web_hints: bool = True

    def __init__(
        self,
        inner: SseTraceSink,
        *,
        workspace: Path,
        thread_id: str,
    ) -> None:
        self._inner = inner
        self._workspace = workspace.expanduser().resolve()
        self._thread_id = thread_id.strip()
        self._step_payloads: list[dict[str, Any]] = []

    @property
    def log_lines(self) -> list[str]:
        return self._inner.log_lines

    def _flush_live(self) -> None:
        if not self._thread_id:
            return
        from llgraph.session.web_trace_store import update_live_web_trace

        update_live_web_trace(
            self._workspace,
            self._thread_id,
            log_lines=list(self._inner.log_lines),
            steps=list(self._step_payloads),
        )

    def line(self, text: str) -> None:
        self._inner.line(text)
        self._flush_live()

    def stream(self, text: str) -> None:
        self._inner.stream(text)

    def stream_end(self) -> None:
        self._inner.stream_end()

    def thinking_update(self, text: str) -> None:
        self._inner.thinking_update(text)

    def step_added(self, step: Any) -> None:
        self._inner.step_added(step)
        self._step_payloads.append(_step_to_dict(step))
        self._flush_live()

    def step_selected(self, step_id: int) -> None:
        self._inner.step_selected(step_id)
