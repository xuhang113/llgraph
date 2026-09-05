"""空记忆库不得触发 embedding 冷启动（首轮 TTFT 白付约 1.5s）。

新工作区的记忆库是空的，但召回原来先 embed 再查表；本地
sentence-transformers 首次加载约 1.5s，全部落在首 token 之前。
向量检索与关键词过滤读同一张表、同一组条件，候选为空时向量必然也为空。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from llgraph.memory.paths import workspace_identity
from llgraph.memory.recall import recall_memories
from llgraph.memory.store import list_memory_rows
from llgraph.memory.write import upsert_memory

pytest.importorskip("lancedb")


@pytest.fixture
def memory_workspace(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("USER", "test-user")
    ws = tmp_path / "proj"
    ws.mkdir()
    (ws / ".llgraph").mkdir()
    return ws


def test_empty_store_skips_embedding(memory_workspace: Path) -> None:
    with patch("llgraph.memory.recall.embed_memory_text") as embed:
        hits, report = recall_memories(memory_workspace, "改一下入口怎么串的")

    assert hits == []
    assert report.hits == []
    embed.assert_not_called()


def test_all_rows_filtered_out_skips_embedding(memory_workspace: Path) -> None:
    """库里有行但都不是 active：同样不该为查空结果加载模型。"""
    with patch("llgraph.memory.write.embed_memory_text", return_value=[0.5] * 8):
        upsert_memory(memory_workspace, content="项目用 uv 管依赖", kind="pref")

    user_id, workspace_key, _ = workspace_identity(memory_workspace)
    rows = list_memory_rows(user_id, workspace_key, status="active")
    assert len(rows) == 1

    with patch(
        "llgraph.memory.recall.list_memory_rows", return_value=[]
    ), patch("llgraph.memory.recall.embed_memory_text") as embed:
        hits, _ = recall_memories(memory_workspace, "依赖怎么装")

    assert hits == []
    embed.assert_not_called()


def test_non_empty_store_still_embeds(memory_workspace: Path) -> None:
    """有候选行时必须照常走向量召回，短路不能吃掉正常路径。"""
    vec = [0.5] * 8
    with patch("llgraph.memory.write.embed_memory_text", return_value=vec):
        upsert_memory(memory_workspace, content="项目用 uv 管依赖，不要用 pip", kind="pref")

    with patch("llgraph.memory.recall.embed_memory_text", return_value=vec) as embed:
        hits, _ = recall_memories(memory_workspace, "依赖怎么装 uv 还是 pip")

    embed.assert_called_once()
    assert [h.content for h in hits] == ["项目用 uv 管依赖，不要用 pip"]
