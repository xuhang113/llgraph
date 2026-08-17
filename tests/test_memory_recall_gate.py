"""记忆召回：相似度门禁，先验不能单独放行。"""

from __future__ import annotations

from llgraph.memory.recall import _passes_similarity_gate, _score_row
from llgraph.memory.settings import MemorySettings, MemoryWeights


def _settings() -> MemorySettings:
    return MemorySettings(
        enabled=True,
        user_id=None,
        auto_recall_top_k=6,
        auto_recall_min_score=0.45,
        recall_min_similarity=0.55,
        recall_min_keyword=0.15,
        search_tool_top_k=10,
        max_inject_tokens=1200,
        memory_content_max_chars=2400,
        memory_inject_excerpt_chars=800,
        scheduled_consolidate_hours=24.0,
        consolidate_on_startup_delay_minutes=5.0,
        max_active_per_workspace=200,
        ttl_never_hit_days=90,
        ttl_no_hit_days=180,
        ttl_low_hit_days=365,
        conflict_cosine_threshold=0.88,
        dedupe_cosine_threshold=0.92,
        weights=MemoryWeights(),
    )


def test_similarity_gate_blocks_low_cosine_without_keywords() -> None:
    assert not _passes_similarity_gate(
        0.42,
        0.0,
        min_similarity=0.55,
        min_keyword=0.15,
    )


def test_similarity_gate_allows_high_cosine() -> None:
    assert _passes_similarity_gate(
        0.87,
        0.0,
        min_similarity=0.55,
        min_keyword=0.15,
    )


def test_similarity_gate_allows_keyword_escape() -> None:
    assert _passes_similarity_gate(
        0.2,
        0.2,
        min_similarity=0.55,
        min_keyword=0.15,
    )


def test_priors_alone_cannot_inflate_past_relevance() -> None:
    """无相似度时融合分应远低于默认 min_score。"""
    row = {
        "kind": "pref",
        "updated_at": "2026-07-14T00:00:00+00:00",
        "confidence": 0.9,
        "hit_count": 4,
        "weight_boost": 0.0,
        "source": "hot_tool",
    }
    score = _score_row(row, cosine=0.0, keyword=0.0, settings=_settings())
    assert score < 0.45
