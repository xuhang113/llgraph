"""本地 embedding 模型：并发只加载一次 + 仅在有记忆可召回时后台预热。"""

from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from llgraph.code_index import local_embedder
from llgraph.code_index.embedding_config import EmbeddingProfile
from llgraph.memory.scheduler import schedule_memory_embedder_prewarm


def _profile(model: str = "test/fake-embed", provider: str = "local") -> EmbeddingProfile:
    return EmbeddingProfile(
        provider=provider,  # type: ignore[arg-type]
        model=model,
        dimension=8,
        batch_size=4,
        device="cpu",
        normalize=True,
        local_files_only=True,
        base_url=None,
        api_key=None,
    )


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    """替换 sentence_transformers，让构造函数可计数、可延迟。"""
    calls: list[str] = []
    gate = threading.Event()

    class _FakeModel:
        def __init__(self, model_id: str, **_kwargs) -> None:
            calls.append(model_id)
            gate.wait(timeout=5.0)

        def get_sentence_embedding_dimension(self) -> int:
            return 8

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(local_embedder, "_MODEL_CACHE", {})
    monkeypatch.setattr(local_embedder, "_MODEL_LOAD_LOCK", threading.Lock())
    yield calls, gate
    gate.set()


def test_concurrent_first_use_loads_model_once(fake_sentence_transformers) -> None:
    calls, gate = fake_sentence_transformers
    profile = _profile()
    results: list[object] = []

    def _load() -> None:
        results.append(local_embedder._get_model(profile))

    threads = [threading.Thread(target=_load) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.05)
    gate.set()
    for t in threads:
        t.join(timeout=10.0)

    assert len(calls) == 1, f"并发首次调用加载了 {len(calls)} 次模型"
    assert len(results) == 4
    assert all(r is results[0] for r in results)


def test_local_embedder_is_loaded_reflects_cache(fake_sentence_transformers) -> None:
    calls, gate = fake_sentence_transformers
    gate.set()
    profile = _profile()
    assert not local_embedder.local_embedder_is_loaded(profile)
    local_embedder._get_model(profile)
    assert local_embedder.local_embedder_is_loaded(profile)


def test_prewarm_skipped_when_memory_store_empty(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("USER", "test-user")
    ws = tmp_path / "proj"
    (ws / ".llgraph").mkdir(parents=True)

    with patch(
        "llgraph.memory.scheduler._memory_store_has_rows", return_value=False
    ), patch("llgraph.code_index.local_embedder.prewarm_local_embedder") as warm:
        assert schedule_memory_embedder_prewarm(ws) is None
    warm.assert_not_called()


def test_prewarm_runs_in_background_when_memories_exist(
    tmp_path: Path, monkeypatch, fake_sentence_transformers
) -> None:
    calls, gate = fake_sentence_transformers
    gate.set()
    monkeypatch.setenv("USER", "test-user")
    ws = tmp_path / "proj"
    (ws / ".llgraph").mkdir(parents=True)

    with patch(
        "llgraph.memory.scheduler._memory_store_has_rows", return_value=True
    ), patch(
        "llgraph.code_index.embedding_config.resolve_embedding_profile",
        return_value=_profile(),
    ):
        thread = schedule_memory_embedder_prewarm(ws)

    assert thread is not None
    assert thread.daemon
    thread.join(timeout=10.0)
    assert calls == ["test/fake-embed"]


def test_prewarm_skipped_for_remote_provider(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("USER", "test-user")
    ws = tmp_path / "proj"
    (ws / ".llgraph").mkdir(parents=True)

    with patch(
        "llgraph.memory.scheduler._memory_store_has_rows", return_value=True
    ), patch(
        "llgraph.code_index.embedding_config.resolve_embedding_profile",
        return_value=_profile(provider="remote"),
    ):
        assert schedule_memory_embedder_prewarm(ws) is None
