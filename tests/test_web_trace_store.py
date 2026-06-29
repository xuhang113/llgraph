"""web_trace_store 多轮 + live 合并测试。"""

from __future__ import annotations

from pathlib import Path

from llgraph.session.web_trace_store import (
    load_last_web_trace,
    load_web_trace_turns,
    save_last_web_trace,
    update_live_web_trace,
)


def test_load_last_web_trace_appends_live_without_double_wrapping(tmp_path: Path) -> None:
    thread_id = "cli-test01"
    save_last_web_trace(
        tmp_path,
        thread_id,
        log_lines=["[10:00:00] 本轮完成"],
        steps=[],
    )
    update_live_web_trace(
        tmp_path,
        thread_id,
        log_lines=["[10:01:00] 准备中...", "[10:01:00] 思考中..."],
        steps=[],
    )
    data = load_last_web_trace(tmp_path, thread_id)
    assert data is not None
    lines = [str(x) for x in data.get("log_lines") or []]
    sep_count = sum(1 for line in lines if line.startswith("─── 本轮"))
    assert sep_count == 2
    assert lines[-2] == "[10:01:00] 准备中..."
    assert lines[-1] == "[10:01:00] 思考中..."


def test_load_web_trace_turns_keeps_per_turn_steps(tmp_path: Path) -> None:
    thread_id = "cli-test03"
    save_last_web_trace(
        tmp_path,
        thread_id,
        log_lines=["line-a"],
        steps=[{"step_id": 1, "kind": "plan", "title": "模型决策", "elapsed": 1.0, "summary": "a"}],
    )
    save_last_web_trace(
        tmp_path,
        thread_id,
        log_lines=["line-b"],
        steps=[{"step_id": 1, "kind": "tool", "title": "执行 grep", "elapsed": 0.1, "summary": "b"}],
    )
    turns = load_web_trace_turns(tmp_path, thread_id)
    assert len(turns) == 2
    assert turns[0]["turn_index"] == 1
    assert turns[1]["turn_index"] == 2
    assert turns[0]["steps"][0]["step_id"] == 1
    assert turns[1]["steps"][0]["step_id"] == 1
    assert turns[0]["label"].startswith("第 1 轮")
    assert turns[1]["label"].startswith("第 2 轮")


def test_load_last_web_trace_live_idempotent(tmp_path: Path) -> None:
    thread_id = "cli-test02"
    update_live_web_trace(
        tmp_path,
        thread_id,
        log_lines=["[10:01:00] 思考中..."],
        steps=[],
    )
    first = load_last_web_trace(tmp_path, thread_id)
    second = load_last_web_trace(tmp_path, thread_id)
    assert first is not None and second is not None
    assert first.get("log_lines") == second.get("log_lines")
