"""Think 回填同 step_id 应 upsert，避免 live Trace 重复多行 Think。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llgraph.console.runtime.sse_sink import PersistingSseTraceSink, SseTraceSink
from llgraph.session.web_trace_store import _dedupe_steps_by_id, load_web_trace_turns, update_live_web_trace


@dataclass
class _FakeStep:
    step_id: int
    kind: str = "thinking"
    title: str = "Think"
    elapsed: float = 14.01
    summary: str = ""
    body_lines: list[str] = field(default_factory=list)
    usage: Any = None
    invoke_timing: Any = None
    elapsed_kind: str = "model"
    sub_thread: str | None = None


def test_persisting_sink_upserts_same_step_id(tmp_path: Path) -> None:
    events: list[dict[str, Any]] = []
    sink = PersistingSseTraceSink(
        SseTraceSink(events.append),
        workspace=tmp_path,
        thread_id="t-upsert",
    )
    sink.step_added(_FakeStep(5, summary="556 字", body_lines=["a" * 556]))
    sink.step_added(_FakeStep(5, summary="615 字", body_lines=["a" * 615]))
    sink.step_added(_FakeStep(6, kind="reply", title="助手回复", summary="ok"))

    assert len(sink._step_payloads) == 2
    assert sink._step_payloads[0]["step_id"] == 5
    assert sink._step_payloads[0]["summary"] == "615 字"
    assert len(sink._step_payloads[0]["body_lines"][0]) == 615
    assert sink._step_payloads[1]["step_id"] == 6


def test_dedupe_steps_keeps_last_body() -> None:
    steps = [
        {"step_id": 1, "summary": "10 字", "body_lines": ["x" * 10]},
        {"step_id": 1, "summary": "20 字", "body_lines": ["x" * 20]},
        {"step_id": 2, "summary": "ok"},
    ]
    out = _dedupe_steps_by_id(steps)
    assert len(out) == 2
    assert out[0]["summary"] == "20 字"
    assert out[1]["step_id"] == 2


def test_load_live_turns_dedupes(tmp_path: Path) -> None:
    update_live_web_trace(
        tmp_path,
        "t-live",
        log_lines=["user"],
        steps=[
            {"step_id": 1, "kind": "thinking", "title": "Think", "summary": "5 字", "body_lines": ["hello"]},
            {"step_id": 1, "kind": "thinking", "title": "Think", "summary": "11 字", "body_lines": ["hello world"]},
        ],
    )
    turns = load_web_trace_turns(tmp_path, "t-live")
    assert len(turns) == 1
    assert len(turns[0]["steps"]) == 1
    assert turns[0]["steps"][0]["summary"] == "11 字"
