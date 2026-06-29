"""execution.jsonl 落盘测试。"""

from __future__ import annotations

import json
from pathlib import Path

from llgraph.display.execution_log import (
    execution_log_path,
    format_execution_record,
    log_turn_failure,
    read_execution_tail,
)


def test_log_turn_failure_appends_turn_error(tmp_path: Path, monkeypatch) -> None:
    llgraph_dir = tmp_path / ".llgraph"
    llgraph_dir.mkdir()
    (llgraph_dir / "agent.json").write_text(
        json.dumps({"logging": {"execution_log": True}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    class _BrokenAgent:
        def get_state(self, _config: dict) -> object:
            raise RuntimeError("no state")

    log_turn_failure(
        tmp_path,
        thread_id="cli-deadbeef",
        with_memory=True,
        agent=_BrokenAgent(),
        duration_sec=12.5,
        error=RuntimeError("boom from gateway"),
        tool_names=["grep_files", "read_file"],
        trace_mode="steps",
    )

    records = read_execution_tail(tmp_path, limit=5)
    assert len(records) == 1
    row = records[0]
    assert row["event"] == "turn_error"
    assert row["outcome"] == "error"
    assert row["thread_id"] == "cli-deadbeef"
    assert row["tools"] == ["grep_files", "read_file"]
    assert row["error_type"] == "RuntimeError"
    assert "boom from gateway" in row["error_message"]
    assert execution_log_path(tmp_path).is_file()


def test_format_execution_record_turn_error() -> None:
    line = format_execution_record(
        {
            "ts": "2026-06-26T03:30:22Z",
            "event": "turn_error",
            "outcome": "error",
            "thread_id": "cli-d8718727",
            "duration_sec": 79.2,
            "usage": {"totals": {"input_tokens": 15252, "output_tokens": 318}},
            "prompt_cache_used": True,
            "tools": ["grep_files"],
            "trace_step_count": 16,
            "last_trace_step_id": 16,
            "error_type": "UnstructuredToolCallError",
            "error_message": "[kimi-k2.6] 模型在正文返回工具 XML",
        },
    )
    assert "turn_error(error)" in line
    assert "cli-d871" in line
    assert "79.2s" in line
    assert "trace_steps=16" in line
    assert "last=#16" in line
    assert "UnstructuredToolCallError" in line


def test_format_execution_record_turn_start() -> None:
    line = format_execution_record(
        {
            "ts": "2026-06-26T03:29:03Z",
            "event": "turn_start",
            "thread_id": "cli-d8718727",
            "model": "kimi-k2.6",
            "user_message_preview": "test 环境接口定位",
        },
    )
    assert "turn_start" in line
    assert "kimi-k2.6" in line
    assert "test 环境" in line


def test_format_execution_record_ts_local_display() -> None:
    from llgraph.display.execution_log import _format_ts_display

    # UTC 08:09 = 北京时间 16:09（UTC+8）
    assert _format_ts_display("2026-06-28T08:09:13Z").endswith("16:09:13")
