"""Web Console 长期记忆检索 API 支撑。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llgraph.memory.paths import (
    ACTIVE_KINDS,
    KIND_FACT,
    KIND_PREF,
    KIND_PROC,
    STATUS_ACTIVE,
    memory_root,
    workspace_identity,
)
from llgraph.memory.recall import recall_memories
from llgraph.memory.settings import resolve_memory_settings
from llgraph.memory.store import list_memory_rows


def _strip_vector(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "vector"}


def _parse_kinds(kind: str | None) -> tuple[str, ...] | None:
    raw = (kind or "").strip().lower()
    if not raw or raw in ("all", "*"):
        return ACTIVE_KINDS
    if raw in ACTIVE_KINDS:
        return (raw,)
    return ACTIVE_KINDS


def memory_status(workspace: Path) -> dict[str, Any]:
    """
    长期记忆库概览（Web 排查用）。

    @param workspace 工作区根
    @return 状态 dict
    """
    settings = resolve_memory_settings(workspace)
    user_id, workspace_key, workspace_slug = workspace_identity(workspace)
    root = memory_root(user_id, workspace_key)
    counts = {"active": 0, KIND_PREF: 0, KIND_FACT: 0, KIND_PROC: 0}
    if settings.enabled:
        active_rows = list_memory_rows(user_id, workspace_key, status=STATUS_ACTIVE)
        counts["active"] = len(active_rows)
        for row in active_rows:
            k = str(row.get("kind", ""))
            if k in counts:
                counts[k] += 1
    return {
        "enabled": settings.enabled,
        "user_id": user_id,
        "workspace_key": workspace_key,
        "workspace_slug": workspace_slug,
        "memory_root": str(root),
        "counts": counts,
        "settings": {
            "auto_recall_top_k": settings.auto_recall_top_k,
            "search_tool_top_k": settings.search_tool_top_k,
            "auto_recall_min_score": settings.auto_recall_min_score,
            "recall_min_similarity": settings.recall_min_similarity,
            "recall_min_keyword": settings.recall_min_keyword,
            "memory_content_max_chars": settings.memory_content_max_chars,
        },
    }


def search_memories_for_web(
    workspace: Path,
    *,
    query: str,
    top_k: int = 20,
    min_score: float = 0.0,
    kind: str | None = None,
) -> dict[str, Any]:
    """
    向量 + 关键词融合检索，或空 query 时浏览列表。

    @param workspace 工作区根
    @param query 检索词；空则按 updated_at 浏览
    @param top_k 条数上限
    @param min_score 最低融合分（排查可设 0）
    @param kind pref/fact/proc/all
    @return API 响应体
    """
    settings = resolve_memory_settings(workspace)
    user_id, workspace_key, _ = workspace_identity(workspace)
    kinds = _parse_kinds(kind)
    q = (query or "").strip()
    cap = max(1, min(top_k, 100))

    if not settings.enabled:
        return {
            "enabled": False,
            "mode": "disabled",
            "query": q,
            "top_k": cap,
            "hits": [],
            "filtered_below_min": 0,
            "elapsed_ms": 0.0,
            "message": "长期记忆未启用（context.memory.enabled=false）",
        }

    if not q:
        rows = list_memory_rows(user_id, workspace_key, status=STATUS_ACTIVE, kinds=kinds)
        rows.sort(key=lambda r: str(r.get("updated_at", "")), reverse=True)
        hits = [_strip_vector(r) for r in rows[:cap]]
        return {
            "enabled": True,
            "mode": "browse",
            "query": "",
            "top_k": cap,
            "hits": hits,
            "filtered_below_min": 0,
            "elapsed_ms": 0.0,
            "count": len(hits),
        }

    mem_hits, report = recall_memories(
        workspace,
        q,
        top_k=cap,
        min_score=min_score,
        for_tool=True,
    )
    rows_by_id = {
        str(r.get("memory_id")): r
        for r in list_memory_rows(user_id, workspace_key, status=STATUS_ACTIVE, kinds=kinds)
    }
    hits: list[dict[str, Any]] = []
    for hit in mem_hits:
        if kinds and hit.kind not in kinds:
            continue
        row = rows_by_id.get(hit.memory_id, {})
        payload = _strip_vector(row) if row else {"memory_id": hit.memory_id, "kind": hit.kind, "content": hit.content}
        payload["score"] = round(hit.score, 4)
        payload["similarity"] = round(hit.similarity, 4)
        payload["hit_count"] = hit.hit_count
        hits.append(payload)

    return {
        "enabled": True,
        "mode": "search",
        "query": q,
        "top_k": cap,
        "min_score": min_score,
        "hits": hits,
        "filtered_below_min": report.filtered_below_min,
        "elapsed_ms": round(report.elapsed_ms, 1),
        "count": len(hits),
    }
