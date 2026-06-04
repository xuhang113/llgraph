"""向量检索执行与可观测日志。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from llgraph.code_index.embedder import embed_query
from llgraph.code_index.store import get_index_status, search_vectors
from llgraph.config.logging_settings import get_search_logger


@dataclass(frozen=True)
class VectorSearchOutcome:
    """向量检索结果与耗时。"""

    hits: list[dict]
    chunk_count: int
    embed_ms: float
    lance_ms: float
    skipped: bool
    skip_reason: str = ""


def perform_vector_search(
    workspace: Path,
    query: str,
    *,
    top_k: int = 15,
    path_prefix: str = ".",
    source: str = "unknown",
    tool: str = "vector",
) -> VectorSearchOutcome:
    """
    执行 LanceDB 向量检索并写 debug/info 日志。

    @param workspace 工作区根
    @param query 查询文本
    @param top_k 返回条数
    @param path_prefix 路径前缀过滤
    @param source 调用来源（agent / cli / hybrid）
    @param tool 工具名（search_code_semantic 等）
    @return VectorSearchOutcome
    """
    logger = get_search_logger()
    prefix = path_prefix.strip() if path_prefix else ""
    status = get_index_status(workspace)

    if not status.exists or status.chunk_count == 0:
        logger.warning(
            "[vector] 跳过：索引为空 source=%s tool=%s path_prefix=%r query_len=%d lance=%s",
            source,
            tool,
            prefix or ".",
            len(query),
            status.lance_path,
        )
        return VectorSearchOutcome(
            hits=[],
            chunk_count=0,
            embed_ms=0.0,
            lance_ms=0.0,
            skipped=True,
            skip_reason="empty_index",
        )

    t_embed = time.perf_counter()
    try:
        qvec = embed_query(workspace, query)
    except Exception as exc:
        logger.error(
            "[vector] embed 失败 source=%s tool=%s error=%s query_len=%d",
            source,
            tool,
            exc,
            len(query),
        )
        raise
    embed_ms = (time.perf_counter() - t_embed) * 1000.0

    t_lance = time.perf_counter()
    hits = search_vectors(
        workspace,
        qvec,
        top_k=top_k,
        path_prefix=prefix,
    )
    lance_ms = (time.perf_counter() - t_lance) * 1000.0

    logger.info(
        "[vector] 完成 source=%s tool=%s hits=%d top_k=%d chunks=%d "
        "embed_ms=%.0f lance_ms=%.0f path_prefix=%r query_len=%d",
        source,
        tool,
        len(hits),
        top_k,
        status.chunk_count,
        embed_ms,
        lance_ms,
        prefix or ".",
        len(query),
    )

    if logger.isEnabledFor(logging.DEBUG):
        preview = query.replace("\n", " ").strip()[:160]
        logger.debug("[vector] query=%r", preview)
        for rank, hit in enumerate(hits[:8], start=1):
            dist = hit.get("_distance", "")
            dist_s = f"{dist:.4f}" if isinstance(dist, (int, float)) else str(dist)
            logger.debug(
                "[vector] hit %d: %s:%s-%s dist=%s",
                rank,
                hit.get("rel_path", ""),
                hit.get("start_line", ""),
                hit.get("end_line", ""),
                dist_s,
            )

    return VectorSearchOutcome(
        hits=hits,
        chunk_count=status.chunk_count,
        embed_ms=embed_ms,
        lance_ms=lance_ms,
        skipped=False,
    )
