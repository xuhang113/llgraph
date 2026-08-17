"""ReAct max_turns 配置解析测试。"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llgraph.core.react_limits import (
    DEFAULT_REACT_MAX_TURNS,
    REACT_MAX_TURNS_CAP,
    format_tool_round_budget_line,
    parse_react_max_turns,
    resolve_agent_max_turns,
)
from llgraph.plan.config import resolve_plan_settings


def test_parse_react_max_turns_defaults_and_clamps() -> None:
    assert DEFAULT_REACT_MAX_TURNS == 100
    assert parse_react_max_turns(None) == DEFAULT_REACT_MAX_TURNS
    assert parse_react_max_turns(120) == 120
    assert parse_react_max_turns(9999) == REACT_MAX_TURNS_CAP
    assert parse_react_max_turns(0) == 1
    assert parse_react_max_turns("bad") == DEFAULT_REACT_MAX_TURNS


def test_resolve_agent_max_turns_from_workspace(tmp_path: Path, monkeypatch) -> None:
    llgraph_dir = tmp_path / ".llgraph"
    llgraph_dir.mkdir()
    (llgraph_dir / "agent.json").write_text(
        json.dumps({"agent": {"max_turns": 80}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    assert resolve_agent_max_turns(tmp_path) == 80


def test_plan_defaults_max_turns_match_agent_default(tmp_path: Path, monkeypatch) -> None:
    llgraph_dir = tmp_path / ".llgraph"
    llgraph_dir.mkdir()
    (llgraph_dir / "agent.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    settings = resolve_plan_settings(tmp_path)
    assert settings.planner_max_turns == DEFAULT_REACT_MAX_TURNS
    assert settings.worker_max_turns == DEFAULT_REACT_MAX_TURNS


def test_format_tool_round_budget_line() -> None:
    msgs = [
        HumanMessage(content="<user_query>\nx\n</user_query>"),
        AIMessage(content="", tool_calls=[{"id": "1", "name": "grep_files", "args": {}}]),
        ToolMessage(content="a", tool_call_id="1", name="grep_files"),
        AIMessage(content="", tool_calls=[{"id": "2", "name": "grep_files", "args": {}}]),
        ToolMessage(content="b", tool_call_id="2", name="grep_files"),
    ]
    line = format_tool_round_budget_line(msgs, workspace=None)
    assert "上限 100" in line
    assert "已用 2" in line
    assert "还能调用约 98" in line
