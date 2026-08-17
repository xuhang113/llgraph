"""长期记忆配置（agent.json context.memory）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llgraph.config.edit_settings import load_agent_config
from llgraph.memory.paths import (
    DEFAULT_CONTENT_MAX_CHARS,
    DEFAULT_INJECT_EXCERPT_CHARS,
)


@dataclass(frozen=True)
class MemoryWeights:
    """召回融合权重。"""

    w_v: float = 1.0
    w_k: float = 0.8
    w_recency: float = 0.2
    w_conf: float = 0.4
    w_use: float = 0.15


@dataclass(frozen=True)
class MemorySettings:
    """长期记忆参数。"""

    enabled: bool
    user_id: str | None
    auto_recall_top_k: int
    auto_recall_min_score: float
    recall_min_similarity: float
    recall_min_keyword: float
    search_tool_top_k: int
    max_inject_tokens: int
    memory_content_max_chars: int
    memory_inject_excerpt_chars: int
    scheduled_consolidate_hours: float
    consolidate_on_startup_delay_minutes: float
    max_active_per_workspace: int
    ttl_never_hit_days: int
    ttl_no_hit_days: int
    ttl_low_hit_days: int
    conflict_cosine_threshold: float
    dedupe_cosine_threshold: float
    weights: MemoryWeights


def _float(val: object, default: float, *, lo: float | None = None, hi: float | None = None) -> float:
    try:
        out = float(val)
    except (TypeError, ValueError):
        return default
    if lo is not None:
        out = max(lo, out)
    if hi is not None:
        out = min(hi, out)
    return out


def _int(val: object, default: int, *, lo: int | None = None, hi: int | None = None) -> int:
    try:
        out = int(val)
    except (TypeError, ValueError):
        return default
    if lo is not None:
        out = max(lo, out)
    if hi is not None:
        out = min(hi, out)
    return out


def _bool(val: object, default: bool) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() not in ("0", "false", "no")
    if val is None:
        return default
    return bool(val)


def resolve_memory_settings(workspace: Path) -> MemorySettings:
    """
    解析 context.memory 配置。

    @param workspace 工作区根
    @return MemorySettings
    """
    cfg = load_agent_config(workspace)
    ctx = cfg.get("context") if isinstance(cfg.get("context"), dict) else {}
    mem = ctx.get("memory") if isinstance(ctx.get("memory"), dict) else {}
    w_raw = mem.get("weights") if isinstance(mem.get("weights"), dict) else {}
    weights = MemoryWeights(
        w_v=_float(w_raw.get("w_v"), 1.0),
        w_k=_float(w_raw.get("w_k"), 0.8),
        w_recency=_float(w_raw.get("w_recency"), 0.2),
        w_conf=_float(w_raw.get("w_conf"), 0.4),
        w_use=_float(w_raw.get("w_use"), 0.15),
    )
    uid = mem.get("user_id")
    user_id = str(uid).strip() if uid else None
    return MemorySettings(
        enabled=_bool(mem.get("enabled"), True),
        user_id=user_id,
        auto_recall_top_k=_int(mem.get("auto_recall_top_k"), 6, lo=1, hi=20),
        # 融合分门槛（在相似度门禁之后）；默认抬高避免先验加成单独过线
        auto_recall_min_score=_float(mem.get("auto_recall_min_score"), 0.45, lo=0.0, hi=2.0),
        # 余弦相似度门禁：不过线且无足够关键词命中则不注入/不返回
        recall_min_similarity=_float(mem.get("recall_min_similarity"), 0.55, lo=0.0, hi=1.0),
        recall_min_keyword=_float(mem.get("recall_min_keyword"), 0.15, lo=0.0, hi=1.0),
        search_tool_top_k=_int(mem.get("search_tool_top_k"), 10, lo=1, hi=30),
        max_inject_tokens=_int(mem.get("max_inject_tokens"), 1200, lo=200, hi=8000),
        memory_content_max_chars=_int(
            mem.get("memory_content_max_chars"), DEFAULT_CONTENT_MAX_CHARS, lo=200, hi=8000
        ),
        memory_inject_excerpt_chars=_int(
            mem.get("memory_inject_excerpt_chars"), DEFAULT_INJECT_EXCERPT_CHARS, lo=80, hi=4000
        ),
        scheduled_consolidate_hours=_float(mem.get("scheduled_consolidate_hours"), 24.0, lo=1.0),
        consolidate_on_startup_delay_minutes=_float(
            mem.get("consolidate_on_startup_delay_minutes"), 5.0, lo=0.0
        ),
        max_active_per_workspace=_int(mem.get("max_active_per_workspace"), 200, lo=20, hi=2000),
        ttl_never_hit_days=_int(mem.get("ttl_never_hit_days"), 90, lo=7),
        ttl_no_hit_days=_int(mem.get("ttl_no_hit_days"), 180, lo=14),
        ttl_low_hit_days=_int(mem.get("ttl_low_hit_days"), 365, lo=30),
        conflict_cosine_threshold=_float(mem.get("conflict_cosine_threshold"), 0.88, lo=0.5, hi=0.99),
        dedupe_cosine_threshold=_float(mem.get("dedupe_cosine_threshold"), 0.92, lo=0.5, hi=0.99),
        weights=weights,
    )
