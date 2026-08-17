"""EventHub 本轮缓冲：高频流式事件不入缓冲，避免挤掉步骤回放。"""

from __future__ import annotations

from llgraph.console.runtime.event_hub import EventHub


def test_thinking_and_stream_not_buffered() -> None:
    hub = EventHub()
    channel = "session:cli-test"
    hub.publish(channel, {"type": "turn_start", "thread_id": "cli-test"})
    for i in range(50):
        hub.publish(channel, {"type": "thinking_delta", "text": f"t{i}"})
        hub.publish(channel, {"type": "stream_delta", "text": f"s{i}"})
        hub.publish(channel, {"type": "trace_activity", "elapsed_sec": i})
    hub.publish(
        channel,
        {"type": "trace_step", "step": {"step_id": 1, "kind": "model", "title": "决策"}},
    )

    q = hub.subscribe(channel)
    replayed = []
    while True:
        try:
            replayed.append(q.get_nowait())
        except Exception:
            break

    types = [str(e.get("type")) for e in replayed]
    assert "turn_start" in types
    assert "trace_step" in types
    assert "thinking_delta" not in types
    assert "stream_delta" not in types
    assert "trace_activity" not in types


def test_structural_events_survive_many_skipped() -> None:
    hub = EventHub()
    channel = "session:cli-long"
    hub.publish(channel, {"type": "turn_start"})
    for i in range(1, 101):
        hub.publish(channel, {"type": "thinking_delta", "text": "x" * 200})
        hub.publish(
            channel,
            {"type": "trace_step", "step": {"step_id": i, "kind": "tool", "title": f"t{i}"}},
        )

    q = hub.subscribe(channel)
    replayed = []
    while True:
        try:
            replayed.append(q.get_nowait())
        except Exception:
            break

    steps = [e for e in replayed if e.get("type") == "trace_step"]
    assert len(steps) == 100
    assert replayed[0].get("type") == "turn_start"
