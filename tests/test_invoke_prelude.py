"""invoke 前 trace 里程碑应写入 trace_sink（Web SSE）。"""

from llgraph.display.trace_display import TraceSession, print_invoke_prelude


class _CaptureSink:
    preserves_ansi = False

    def __init__(self) -> None:
        self.lines: list[str] = []

    def line(self, text: str) -> None:
        self.lines.append(text)


def test_print_invoke_prelude_uses_trace_sink():
    sink = _CaptureSink()
    trace = TraceSession()
    trace.trace_sink = sink
    print_invoke_prelude(trace)
    assert len(sink.lines) == 1
    assert "准备中" in sink.lines[0]
    assert "压缩上下文" in sink.lines[0]
