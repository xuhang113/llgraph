"""空记忆库不得触发冷启动：既不加载 embedding 模型，也不 import lancedb。

新工作区的记忆库是空的，但召回原来先 embed 再查表：本地
sentence-transformers 首次加载约 1.5s、`import lancedb` 约 0.8s，
两笔都落在首 token 之前，换来的只是「表里没有行」。
向量检索与关键词过滤读同一张表、同一组条件，候选为空时向量必然也为空。
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from llgraph.memory.paths import memory_store_is_definitely_empty, workspace_identity
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


def test_empty_store_does_not_open_lancedb(memory_workspace: Path) -> None:
    """目录还不存在时连 connect 都不该发生（connect 会 import lancedb 并建目录）。"""
    user_id, workspace_key, _ = workspace_identity(memory_workspace)
    assert memory_store_is_definitely_empty(user_id, workspace_key)

    with patch("llgraph.memory.store.connect_memory_db") as connect:
        assert list_memory_rows(user_id, workspace_key, status="active") == []
    connect.assert_not_called()


def test_store_with_rows_is_not_reported_empty(memory_workspace: Path) -> None:
    with patch("llgraph.memory.write.embed_memory_text", return_value=[0.5] * 8):
        upsert_memory(memory_workspace, content="项目用 uv 管依赖", kind="pref")

    user_id, workspace_key, _ = workspace_identity(memory_workspace)
    assert not memory_store_is_definitely_empty(user_id, workspace_key)
    assert len(list_memory_rows(user_id, workspace_key, status="active")) == 1


def test_fresh_workspace_recall_never_imports_lancedb(tmp_path: Path) -> None:
    """子进程验真：全新工作区跑一次召回后，lancedb / torch 都不该进 sys.modules。"""
    home = tmp_path / "home"
    ws = tmp_path / "proj"
    (ws / ".llgraph").mkdir(parents=True)
    script = textwrap.dedent(
        f"""
        import os, sys
        os.environ["LLGRAPH_HOME"] = {str(home)!r}
        os.environ["USER"] = "cold-user"
        from pathlib import Path
        from llgraph.memory.recall import build_agent_memories_for_turn

        block, report = build_agent_memories_for_turn(Path({str(ws)!r}), "接着上面的改一下入口")
        assert block == "" and report.hits == [], (block, report.hits)
        heavy = [m for m in ("lancedb", "torch", "sentence_transformers") if m in sys.modules]
        print("HEAVY=" + ",".join(heavy))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "HEAVY=\n" in proc.stdout or proc.stdout.strip().endswith("HEAVY="), (
        f"空库召回加载了重依赖: {proc.stdout.strip()}"
    )
