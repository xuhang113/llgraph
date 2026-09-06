"""长期记忆：批量删除与基本 CRUD。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from llgraph.memory.memory_write_session import MemoryWriteSession
from llgraph.memory.paths import workspace_identity
from llgraph.memory.store import list_memory_rows
from llgraph.memory.write import delete_memory, upsert_memory

# 记忆落盘走 LanceDB，属可选依赖：pip install 'llgraph[index]'
pytest.importorskip("lancedb", reason="需要可选依赖 lancedb")


@pytest.fixture
def memory_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "proj"
    ws.mkdir()
    (ws / ".llgraph").mkdir()
    return ws


def test_batch_delete_memory_ids(memory_workspace: Path, monkeypatch) -> None:
    monkeypatch.setenv("USER", "test-user")
    vectors = [[0.1] * 8, [0.2] * 8, [0.3] * 8]

    def fake_embed(_ws: Path, text: str) -> list[float]:
        idx = {"a": 0, "b": 1, "c": 2}.get(text.strip()[0], 0)
        return vectors[idx]

    with patch("llgraph.memory.write.embed_memory_text", side_effect=fake_embed):
        upsert_memory(memory_workspace, content="alpha pref", kind="pref")
        upsert_memory(memory_workspace, content="beta fact", kind="fact")
        upsert_memory(memory_workspace, content="gamma proc", kind="proc")

    user_id, workspace_key, _ = workspace_identity(memory_workspace)
    rows = list_memory_rows(user_id, workspace_key, status="active")
    assert len(rows) == 3
    ids = [str(r["memory_id"]) for r in rows]
    session = MemoryWriteSession(user_id, workspace_key)
    deleted = session.delete_memory_ids(ids)
    assert deleted == 3
    assert list_memory_rows(user_id, workspace_key, status="active") == []


def test_upsert_and_delete(memory_workspace: Path, monkeypatch) -> None:
    monkeypatch.setenv("USER", "test-user")

    with patch("llgraph.memory.write.embed_memory_text", return_value=[0.5] * 8):
        report = upsert_memory(memory_workspace, content="使用简体中文回复", kind="pref")
    assert report.action in ("upsert", "replace")
    assert report.memory_id

    user_id, workspace_key, _ = workspace_identity(memory_workspace)
    rows = list_memory_rows(user_id, workspace_key, status="active")
    assert len(rows) == 1
    assert "简体中文" in str(rows[0].get("content", ""))

    delete_memory(memory_workspace, report.memory_id)
    assert list_memory_rows(user_id, workspace_key, status="active") == []
