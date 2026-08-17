"""Web 长期记忆检索 API。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from llgraph.memory.paths import workspace_identity
from llgraph.memory.web_search import memory_status, search_memories_for_web
from llgraph.memory.write import upsert_memory


@pytest.fixture
def memory_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "proj"
    ws.mkdir()
    (ws / ".llgraph").mkdir()
    return ws


def test_memory_status_disabled_counts(memory_workspace: Path, monkeypatch) -> None:
    monkeypatch.setenv("USER", "web-user")
    (memory_workspace / ".llgraph" / "agent.json").write_text(
        '{"context":{"memory":{"enabled":false}}}',
        encoding="utf-8",
    )
    st = memory_status(memory_workspace)
    assert st["enabled"] is False
    assert st["counts"]["active"] == 0


def test_search_browse_and_query(memory_workspace: Path, monkeypatch) -> None:
    monkeypatch.setenv("USER", "web-user")
    with patch("llgraph.memory.write.embed_memory_text", return_value=[0.5] * 8):
        upsert_memory(memory_workspace, content="使用简体中文回复", kind="pref")

    browse = search_memories_for_web(memory_workspace, query="", top_k=10)
    assert browse["mode"] == "browse"
    assert browse["count"] == 1
    assert "简体中文" in browse["hits"][0]["content"]

    search = search_memories_for_web(
        memory_workspace, query="简体中文", top_k=5, min_score=0.0
    )
    assert search["mode"] == "search"
    assert search["count"] >= 1
    assert search["hits"][0].get("score") is not None

    user_id, workspace_key, _ = workspace_identity(memory_workspace)
    assert search_memories_for_web(memory_workspace, query="x")["enabled"] is True
    assert user_id == "web-user"
    assert workspace_key
