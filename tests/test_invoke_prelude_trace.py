"""invoke 前预处理 trace 步骤。"""

from __future__ import annotations

import logging

from llgraph.display.trace_display import TraceSession, TurnTracePrinter
from llgraph.display.trace_emit import emit_explore_trace_step, emit_invoke_prelude_step


class _CaptureSink:
    preserves_ansi = False

    def __init__(self) -> None:
        self.steps: list[dict] = []

    def step_added(self, step) -> None:
        from dataclasses import asdict

        self.steps.append(asdict(step) if hasattr(step, "__dataclass_fields__") else step)


def test_emit_invoke_prelude_step_logs_only_no_sse(caplog) -> None:
    sink = _CaptureSink()
    trace = TraceSession()
    trace.trace_sink = sink

    with caplog.at_level(logging.INFO, logger="llgraph.trace.prelude"):
        emit_invoke_prelude_step(
            trace,
            title="上下文检查",
            summary="未触发",
            elapsed=0.12,
            kind="compress",
        )
        emit_invoke_prelude_step(
            trace,
            title="invoke 准备合计",
            summary="共 1 步",
            elapsed=0.35,
        )

    assert len(trace.pending_invoke_steps) == 0
    assert len(sink.steps) == 0
    assert any("上下文检查" in r.message for r in caplog.records)


def test_adopt_prelude_steps_merges_into_printer() -> None:
    from llgraph.display.trace_display import TraceStepRecord

    trace = TraceSession()
    trace.pending_invoke_steps = [
        TraceStepRecord(
            step_id=1,
            kind="compress",
            title="上下文检查",
            elapsed=0.1,
            summary="未触发",
            elapsed_kind="preprocess",
        )
    ]
    printer = TurnTracePrinter(trace)
    printer.adopt_prelude_steps(list(trace.pending_invoke_steps))
    trace.pending_invoke_steps = []

    assert len(printer._steps) == 1
    assert printer._step_index == 1


def test_emit_explore_trace_step_pushes_sub_thread() -> None:
    sink = _CaptureSink()
    trace = TraceSession()
    trace.trace_sink = sink
    emit_explore_trace_step(
        trace,
        title="Explore",
        summary="auth 相关定位",
        elapsed=1.25,
        sub_thread="cli-abc:explore:a1b2",
        body_lines=["找到 3 个文件"],
    )
    assert len(sink.steps) == 1
    assert sink.steps[0]["kind"] == "explore"
    assert sink.steps[0]["sub_thread"] == "cli-abc:explore:a1b2"
