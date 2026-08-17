"""无 active_printer 时记忆写入应挂到 last_turn_steps。"""

from __future__ import annotations

from llgraph.display.trace_display import TraceMode, TraceSession
from llgraph.display.trace_emit import emit_turn_epilogue_step
from llgraph.memory.trace_emit import emit_memory_write_trace_step
from llgraph.memory.write import MemoryWriteReport


def test_emit_turn_epilogue_appends_last_turn_steps() -> None:
    session = TraceSession(mode=TraceMode.STEPS)
    session.last_turn_steps = []
    session.active_printer = None
    sid = emit_turn_epilogue_step(
        session,
        title="长期记忆 · 写入",
        summary="upsert · [proc] abcd…",
        elapsed=0.1,
        body_lines=["action: upsert"],
        kind="memory_write",
    )
    assert sid == 1
    assert len(session.last_turn_steps) == 1
    assert session.last_turn_steps[0].kind == "memory_write"
    assert session.pending_invoke_steps == []


def test_emit_memory_write_without_printer_uses_epilogue() -> None:
    session = TraceSession(mode=TraceMode.STEPS)
    session.last_turn_steps = []
    session.active_printer = None
    report = MemoryWriteReport(
        action="upsert",
        memory_id="abcd1234-xxxx",
        kind="proc",
        content_preview="配置说明须写清使用场景",
        replaced_ids=[],
        reason="user_explicit",
    )
    emit_memory_write_trace_step(report, session=session)
    assert len(session.last_turn_steps) == 1
    assert session.last_turn_steps[0].title == "长期记忆 · 写入"
