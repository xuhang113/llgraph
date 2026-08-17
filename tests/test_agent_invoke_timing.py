"""Agent LLM 分段计时。"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from llgraph.core.agent_invoke_timing import (
    AgentInvokeTiming,
    attach_invoke_timing,
    read_invoke_timing,
)


def test_attach_and_read_invoke_timing() -> None:
    msg = AIMessage(content="ok")
    timing = AgentInvokeTiming(
        request_id="req-test-1",
        prepare_sec=0.042,
        http_sec=12.5,
        normalize_sec=0.001,
    )
    patched = attach_invoke_timing(msg, timing)
    restored = read_invoke_timing(patched)
    assert restored is not None
    assert restored.request_id == "req-test-1"
    assert abs(restored.prepare_sec - 0.042) < 1e-6
    assert abs(restored.http_sec - 12.5) < 1e-6


def test_mark_agent_step_start_resets_timer() -> None:
    import time

    from llgraph.display.trace_display import TraceSession, TurnTracePrinter

    printer = TurnTracePrinter(TraceSession())
    printer._step_start = time.perf_counter() - 5.0
    printer.mark_agent_step_start()
    elapsed = time.perf_counter() - printer._step_start
    assert elapsed < 0.2
