"""Subagent 共享引擎基础测试。"""

from __future__ import annotations

import json
from pathlib import Path

from llgraph.subagent.profile import get_subagent_profile
from llgraph.subagent.registry import load_subagent_children, register_subagent_child
from llgraph.subagent.result import SubagentResult


def test_explore_profile_thread_suffix():
    prof = get_subagent_profile("explore")
    assert prof.kind == "explore"
    assert prof.allow_write is False
    assert prof.format_sub_thread("cli-abc", "a1b2") == "cli-abc:explore:a1b2"


def test_worker_profile_thread_suffix():
    prof = get_subagent_profile("worker")
    assert prof.format_sub_thread("plan-xyz", "w1") == "plan-xyz:worker:w1"


def test_subagent_result_tool_output():
    result = SubagentResult(
        sub_thread="cli-1:explore:aa",
        kind="explore",
        sub_id="aa",
        summary="找到 auth 在 foo.py",
        status="ok",
        files_changed=[],
    )
    text = result.as_tool_output()
    assert "explore" in text
    assert "cli-1:explore:aa" in text
    assert "foo.py" in text


def test_register_and_load_subagent_children(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    # session_thread_dir 依赖 ~/.llgraph context slug；用 monkeypatch 指向 tmp
    from llgraph.session import user_storage

    sessions = tmp_path / "sessions"
    sessions.mkdir()

    def _fake_session_thread_dir(_ws: Path, thread_id: str) -> Path:
        safe = thread_id.replace("/", "_").strip() or "default"
        return sessions / safe

    monkeypatch.setattr(user_storage, "session_thread_dir", _fake_session_thread_dir)
    monkeypatch.setattr(
        "llgraph.subagent.registry.session_thread_dir",
        _fake_session_thread_dir,
    )

    parent = "cli-test01"
    register_subagent_child(
        workspace,
        parent,
        {
            "kind": "explore",
            "sub_id": "ab12",
            "sub_thread": f"{parent}:explore:ab12",
            "title": "Explore auth",
            "status": "running",
        },
    )
    children = load_subagent_children(workspace, parent)
    assert len(children) == 1
    assert children[0]["status"] == "running"

    register_subagent_child(
        workspace,
        parent,
        {
            "kind": "explore",
            "sub_id": "ab12",
            "sub_thread": f"{parent}:explore:ab12",
            "title": "Explore auth",
            "status": "ok",
        },
    )
    children = load_subagent_children(workspace, parent)
    assert len(children) == 1
    assert children[0]["status"] == "ok"
    assert children[0]["sub_thread"] == f"{parent}:explore:ab12"
    path = sessions / parent / "subagents.json"
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["children"]) == 1


def test_update_explore_trace_step_mutates_same_id():
    from llgraph.display.trace_display import TraceSession
    from llgraph.display.trace_emit import (
        emit_explore_trace_step,
        update_explore_trace_step,
    )

    session = TraceSession(mode="steps")
    added: list = []

    class _Sink:
        def step_added(self, step):
            added.append(step)

    session.trace_sink = _Sink()
    session.last_turn_steps = []
    step_id = emit_explore_trace_step(
        session,
        title="Explore",
        summary="执行中…",
        elapsed=0.0,
        sub_thread="cli-1:explore:aa",
        body_lines=["status=running"],
    )
    assert step_id > 0
    assert len(added) == 1
    assert added[0].summary == "执行中…"

    ok = update_explore_trace_step(
        session,
        step_id,
        summary="找到 auth",
        elapsed=1.25,
        body_lines=["status=ok", "找到 auth"],
    )
    assert ok is True
    assert len(added) == 2
    assert added[1].step_id == step_id
    assert added[1].summary == "找到 auth"
    assert added[1].elapsed == 1.25
    assert session.last_turn_steps[0].summary == "找到 auth"