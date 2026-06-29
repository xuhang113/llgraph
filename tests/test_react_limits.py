"""ReAct max_turns 配置解析测试。"""

from __future__ import annotations

import json
from pathlib import Path

from llgraph.core.react_limits import (
    DEFAULT_REACT_MAX_TURNS,
    parse_react_max_turns,
    resolve_agent_max_turns,
)
from llgraph.plan.config import resolve_plan_settings


def test_parse_react_max_turns_defaults_and_clamps() -> None:
    assert parse_react_max_turns(None) == DEFAULT_REACT_MAX_TURNS
    assert parse_react_max_turns(120) == 120
    assert parse_react_max_turns(9999) == 500
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


def test_plan_defaults_max_turns_500(tmp_path: Path, monkeypatch) -> None:
    llgraph_dir = tmp_path / ".llgraph"
    llgraph_dir.mkdir()
    (llgraph_dir / "agent.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    settings = resolve_plan_settings(tmp_path)
    assert settings.planner_max_turns == 500
    assert settings.worker_max_turns == 500
