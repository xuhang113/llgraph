"""session last_run / run_log 落盘测试。"""

from __future__ import annotations

import json
from pathlib import Path

from llgraph.session.session_run_log import (
    UserCancelledError,
    last_run_path,
    read_session_last_run,
    run_log_path,
    trace_run_context,
    write_session_last_run,
)


class _FakeStep:
    def __init__(self, step_id: int, summary: str) -> None:
        self.step_id = step_id
        self.summary = summary


class _FakeTrace:
    def __init__(self) -> None:
        self.last_turn_steps = [_FakeStep(16, "grep_files getUserInfoByDing")]
        self.active_printer = None


def test_write_session_last_run_and_read(tmp_path: Path, monkeypatch) -> None:
    llgraph_dir = tmp_path / ".llgraph"
    llgraph_dir.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))

    ctx = trace_run_context(_FakeTrace())
    assert ctx["trace_step_count"] == 1
    assert ctx["last_trace_step_id"] == 16

    payload = write_session_last_run(
        tmp_path,
        "cli-deadbeef",
        outcome="cancelled",
        duration_sec=79.2,
        model="kimi-k2.6",
        user_message="test 环境 /api/login/getUserInfoByDing",
        error=UserCancelledError("用户停止当前生成"),
        trace_context=ctx,
        source="web",
    )
    assert payload["outcome"] == "cancelled"
    assert last_run_path(tmp_path, "cli-deadbeef").is_file()
    assert run_log_path(tmp_path, "cli-deadbeef").is_file()

    loaded = read_session_last_run(tmp_path, "cli-deadbeef")
    assert loaded is not None
    assert loaded["trace_step_count"] == 1
    assert loaded["last_trace_step_id"] == 16
    assert "getUserInfoByDing" in loaded["user_message_preview"]
    assert loaded["error_type"] == "UserCancelledError"


def test_run_log_jsonl_appends(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    write_session_last_run(
        tmp_path,
        "cli-abc",
        outcome="ok",
        duration_sec=1.0,
        source="cli",
    )
    write_session_last_run(
        tmp_path,
        "cli-abc",
        outcome="error",
        duration_sec=2.0,
        error=RuntimeError("boom"),
        source="cli",
    )
    lines = run_log_path(tmp_path, "cli-abc").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    second = json.loads(lines[1])
    assert second["outcome"] == "error"
