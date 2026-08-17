"""LLGRAPH_HOME 隔离：单测不得写入真实 ~/.llgraph/context。"""

from __future__ import annotations

from pathlib import Path

from llgraph.core.agent_config import user_llgraph_home
from llgraph.session.user_storage import user_context_root, workspace_context_dir


def test_user_llgraph_home_respects_env(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "custom-home"
    monkeypatch.setenv("LLGRAPH_HOME", str(home))
    assert user_llgraph_home() == home.resolve()
    assert user_context_root() == home.resolve() / "context"


def test_workspace_context_dir_stays_under_llgraph_home(
    monkeypatch, tmp_path: Path
) -> None:
    """模拟 pytest tmp 工作区落盘时，只写到 LLGRAPH_HOME，不碰 ~/.llgraph。"""
    ll_home = tmp_path / "ll-home"
    ws = tmp_path / "pytest-ws"
    ws.mkdir()
    monkeypatch.setenv("LLGRAPH_HOME", str(ll_home))

    real_home_context = Path.home() / ".llgraph" / "context"
    before = set()
    if real_home_context.is_dir():
        before = {p.name for p in real_home_context.iterdir()}

    ctx = workspace_context_dir(ws)
    assert ctx.is_relative_to(ll_home.resolve())
    assert (ctx / "workspace.json").is_file()

    after = set()
    if real_home_context.is_dir():
        after = {p.name for p in real_home_context.iterdir()}
    assert after == before
