"""长期记忆召回、打分与注入块。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from llgraph.cli.search_terms import build_search_terms
from llgraph.memory.embed import embed_memory_text
from llgraph.memory.memory_write_session import MemoryWriteSession
from llgraph.memory.paths import ACTIVE_KINDS, workspace_identity
from llgraph.memory.settings import MemorySettings, resolve_memory_settings
from llgraph.memory.store import (
    _memory_record,
    cosine_similarity,
    list_memory_rows,
    search_memory_vectors,
    utc_now_iso,
)


@dataclass
class MemoryHit:
    """召回命中。"""

    memory_id: str
    kind: str
    content: str
    score: float
    hit_count: int
    similarity: float = 0.0


@dataclass
class MemoryRecallReport:
    """召回报告（trace 用）。"""

    query: str
    top_k: int
    hits: list[MemoryHit] = field(default_factory=list)
    filtered_below_min: int = 0
    elapsed_ms: float = 0.0
    inject_chars: int = 0


def _type_boost(kind: str) -> float:
    return {
        "pref": 0.08,
        "proc": 0.05,
        "fact": 0.04,
    }.get(kind, 0.0)


def _source_boost(source: str) -> float:
    if source == "hot_tool":
        return 0.03
    return 0.0


def _recency_boost(updated_at: str, w: float, tau_days: float = 30.0) -> float:
    if not updated_at:
        return 0.0
    try:
        dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
        return w * math.exp(-max(0.0, delta) / tau_days)
    except (TypeError, ValueError):
        return 0.0


def _keyword_score(content: str, terms: list[str]) -> float:
    if not terms:
        return 0.0
    lower = (content or "").lower()
    hits = sum(1 for t in terms if t.lower() in lower)
    return hits / max(1, len(terms))


def _passes_similarity_gate(
    cosine: float,
    keyword: float,
    *,
    min_similarity: float,
    min_keyword: float,
) -> bool:
    """须靠向量或关键词相关性过线；先验加成不能单独放行。"""
    return cosine >= min_similarity or keyword >= min_keyword


def _score_row(
    row: dict,
    *,
    cosine: float,
    keyword: float,
    settings: MemorySettings,
) -> float:
    """
    融合分：以相似度/关键词为主，先验仅微调排序。

    @param cosine 查询与记忆向量余弦
    @param keyword 关键词命中比率
    """
    w = settings.weights
    score = w.w_v * max(0.0, cosine) + w.w_k * max(0.0, keyword)
    kind = str(row.get("kind", ""))
    score += _type_boost(kind)
    score += _recency_boost(str(row.get("updated_at", "")), w.w_recency * 0.5)
    score += w.w_conf * 0.15 * float(row.get("confidence", 0) or 0)
    hit_count = int(row.get("hit_count", 0) or 0)
    score += w.w_use * 0.35 * math.log1p(hit_count)
    score += float(row.get("weight_boost", 0) or 0)
    score += _source_boost(str(row.get("source", "")))
    return score


def recall_memories(
    workspace: Path,
    query: str,
    *,
    top_k: int | None = None,
    min_score: float | None = None,
    for_tool: bool = False,
) -> tuple[list[MemoryHit], MemoryRecallReport]:
    """
    双路召回并打分（相似度门禁后再融合排序）。

    @param workspace 工作区根
    @param query 检索问句
    @param top_k 条数
    @param min_score 融合分最低（默认用 settings.auto_recall_min_score）
    @param for_tool search_memory 工具
    @return (hits, report)
    """
    t0 = time.perf_counter()
    settings = resolve_memory_settings(workspace)
    k = top_k or (settings.search_tool_top_k if for_tool else settings.auto_recall_top_k)
    threshold = min_score if min_score is not None else settings.auto_recall_min_score
    q = (query or "").strip()
    report = MemoryRecallReport(query=q, top_k=k)

    if not settings.enabled or not q:
        report.elapsed_ms = (time.perf_counter() - t0) * 1000
        return [], report

    user_id, workspace_key, _ = workspace_identity(workspace)
    kinds = ACTIVE_KINDS

    # 先读候选行：向量检索与此处过滤同表同条件（user/workspace/status=active/kinds），
    # 候选为空时向量检索必然也为空，无需为此加载 embedding 模型（本地冷启动约 1.5s）。
    keyword_rows = list_memory_rows(user_id, workspace_key, status="active", kinds=kinds)
    if not keyword_rows:
        report.elapsed_ms = (time.perf_counter() - t0) * 1000
        return [], report

    qvec: list[float] = []
    vector_hits: list[dict] = []
    try:
        qvec = embed_memory_text(workspace, q) or []
        if qvec:
            vector_hits = search_memory_vectors(
                user_id, workspace_key, qvec, top_k=20, kinds=kinds
            )
    except Exception:
        vector_hits = []
        qvec = []

    terms = build_search_terms(topic=q, keywords=q) or [q]
    keyword_scored = sorted(
        keyword_rows,
        key=lambda r: _keyword_score(str(r.get("content", "")), terms),
        reverse=True,
    )[:20]

    by_id: dict[str, dict] = {}
    for row in vector_hits + keyword_scored:
        mid = str(row.get("memory_id", ""))
        if mid and mid not in by_id:
            by_id[mid] = row

    scored: list[tuple[float, float, dict]] = []
    for mid, row in by_id.items():
        if str(row.get("kind", "")) not in ACTIVE_KINDS:
            continue
        old_vec = row.get("vector") or []
        cosine = (
            cosine_similarity(qvec, list(old_vec))
            if qvec and isinstance(old_vec, list) and old_vec
            else 0.0
        )
        keyword = _keyword_score(str(row.get("content", "")), terms)
        if not _passes_similarity_gate(
            cosine,
            keyword,
            min_similarity=settings.recall_min_similarity,
            min_keyword=settings.recall_min_keyword,
        ):
            report.filtered_below_min += 1
            continue
        s = _score_row(row, cosine=cosine, keyword=keyword, settings=settings)
        if s >= threshold:
            scored.append((s, cosine, row))
        else:
            report.filtered_below_min += 1

    scored.sort(key=lambda x: -x[0])
    hits: list[MemoryHit] = []
    for s, cosine, row in scored[:k]:
        hits.append(
            MemoryHit(
                memory_id=str(row.get("memory_id", "")),
                kind=str(row.get("kind", "")),
                content=str(row.get("content", "")),
                score=s,
                hit_count=int(row.get("hit_count", 0) or 0),
                similarity=cosine,
            )
        )

    report.hits = hits
    report.elapsed_ms = (time.perf_counter() - t0) * 1000
    return hits, report


def _excerpt(content: str, max_chars: int) -> str:
    text = (content or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def format_agent_memories_block(
    hits: list[MemoryHit],
    *,
    excerpt_chars: int,
    max_tokens: int,
) -> str:
    """
    构建 <agent-memories> 注入块。

    @param hits 命中列表
    @param excerpt_chars 单条 excerpt
    @param max_tokens 粗略字符预算（×4）
    @return 块文本；无命中为空
    """
    if not hits:
        return ""
    budget = max(200, max_tokens * 4)
    lines = ["<agent-memories>"]
    used = len(lines[0]) + 1
    for hit in hits:
        line = f"- [{hit.kind}] {_excerpt(hit.content, excerpt_chars)}"
        if used + len(line) + 1 > budget:
            break
        lines.append(line)
        used += len(line) + 1
    lines.append("</agent-memories>")
    return "\n".join(lines)


def record_memory_hits(workspace: Path, hits: list[MemoryHit]) -> None:
    """
    更新 last_hit_at / hit_count。

    @param workspace 工作区根
    @param hits 命中列表
    """
    if not hits:
        return
    settings = resolve_memory_settings(workspace)
    if not settings.enabled:
        return
    user_id, workspace_key, workspace_slug = workspace_identity(workspace)
    now = utc_now_iso()
    rows_by_id = {
        str(r.get("memory_id")): r
        for r in list_memory_rows(user_id, workspace_key, status="active")
    }
    session = MemoryWriteSession(user_id, workspace_key)
    to_delete: list[str] = []
    to_add: list[dict] = []
    for hit in hits:
        row = rows_by_id.get(hit.memory_id)
        if not row:
            continue
        to_delete.append(hit.memory_id)
        vec = row.get("vector") or []
        to_add.append(
            _memory_record(
                memory_id=hit.memory_id,
                user_id=user_id,
                workspace_key=workspace_key,
                workspace_slug=workspace_slug,
                kind=str(row.get("kind", "")),
                content=str(row.get("content", "")),
                content_hash=str(row.get("content_hash", "")),
                vector=list(vec) if isinstance(vec, list) else [],
                confidence=float(row.get("confidence", 0) or 0),
                source=str(row.get("source", "")),
                status="active",
                supersedes_id=str(row.get("supersedes_id", "")),
                created_at=str(row.get("created_at", now)),
                updated_at=str(row.get("updated_at", now)),
                last_hit_at=now,
                hit_count=int(row.get("hit_count", 0) or 0) + 1,
                weight_boost=float(row.get("weight_boost", 0) or 0),
                tags=str(row.get("tags", "[]")),
            )
        )
    if to_delete:
        session.delete_memory_ids(to_delete)
    if to_add:
        session.add_records(to_add)


def build_agent_memories_for_turn(
    workspace: Path,
    user_message: str,
    *,
    trace: object | None = None,
) -> tuple[str, MemoryRecallReport]:
    """
    invoke 前自动召回并格式化注入块。

    @param workspace 工作区根
    @param user_message 用户消息
    @param trace TraceSession
    @return (block, report)
    """
    from llgraph.context.context_continuity import strip_workspace_context_wrapper

    settings = resolve_memory_settings(workspace)
    query = strip_workspace_context_wrapper((user_message or "").strip())
    hits, report = recall_memories(workspace, query)
    block = format_agent_memories_block(
        hits,
        excerpt_chars=settings.memory_inject_excerpt_chars,
        max_tokens=settings.max_inject_tokens,
    )
    report.inject_chars = len(block)
    if hits:
        record_memory_hits(workspace, hits)
    if block and trace is not None:
        from llgraph.memory.trace_emit import emit_memory_recall_trace_step

        emit_memory_recall_trace_step(report, session=trace)
    return block, report
