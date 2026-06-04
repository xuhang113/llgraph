"""Hybrid 检索：grep + 向量，RRF 融合。"""

from __future__ import annotations

import re
from pathlib import Path

from llgraph.code_index.vector_search import perform_vector_search
from llgraph.config.logging_settings import get_search_logger
from llgraph.core.workspace import WorkspaceContext

RRF_K = 60
GREP_TOP = 50
VECTOR_TOP = 50


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)


def _grep_hits(
    ctx: WorkspaceContext,
    query: str,
    path_prefix: str,
    *,
    limit: int = GREP_TOP,
) -> list[tuple[str, str]]:
    """
    稀疏检索：提取 query 中的标识符做子串 grep。

    @return (doc_id, snippet) 列表，doc_id 为 rel_path:start_line
    """
    tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z_][\w.-]{2,}", query)
    if not tokens:
        tokens = [query.strip()[:40]]
    tokens = tokens[:8]

    results: list[tuple[str, str]] = []
    seen: set[str] = set()

    for token in tokens:
        try:
            pattern = re.compile(re.escape(token), re.IGNORECASE)
        except re.error:
            continue
        for rel in ctx.iter_files(path_prefix or "."):
            full = ctx.resolve_path(rel)
            if not full.is_file():
                continue
            try:
                if full.stat().st_size > 500_000:
                    continue
                text = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    doc_id = f"{rel}:{line_no}"
                    if doc_id in seen:
                        continue
                    seen.add(doc_id)
                    snippet = line.strip()[:160]
                    results.append((doc_id, snippet))
                    if len(results) >= limit:
                        return results
    return results


def _vector_hit_ids(
    workspace: Path,
    query: str,
    path_prefix: str,
    top_k: int,
    *,
    source: str = "cli",
    tool: str = "search_code_hybrid",
) -> tuple[list[tuple[str, str]], str]:
    """
    向量路：返回 (doc_id, preview) 与错误信息。

    @param workspace 工作区根
    @param query 查询
    @param path_prefix 路径前缀
    @param top_k 条数
    @param source 调用来源
    @param tool 工具名
    @return (命中列表, 错误文案；无错为空串)
    """
    logger = get_search_logger()
    try:
        outcome = perform_vector_search(
            workspace,
            query,
            top_k=top_k,
            path_prefix=path_prefix,
            source=source,
            tool=tool,
        )
    except Exception as exc:
        logger.error(
            "[vector] hybrid 向量路失败 source=%s tool=%s error=%s",
            source,
            tool,
            exc,
        )
        return [], str(exc)

    if outcome.skipped:
        return [], "索引为空"

    out: list[tuple[str, str]] = []
    for hit in outcome.hits:
        rel = hit.get("rel_path", "")
        start = hit.get("start_line", 0)
        doc_id = f"{rel}:{start}"
        preview = hit.get("text_preview", "")
        out.append((doc_id, preview))
    return out, ""


def search_hybrid(
    workspace: Path,
    query: str,
    *,
    top_k: int = 10,
    path_prefix: str = ".",
    source: str = "cli",
    tool: str = "search_code_hybrid",
) -> str:
    """
    RRF 融合 grep 与向量检索。

    @param workspace 工作区根
    @param query 查询
    @param top_k 最终条数
    @param path_prefix 路径前缀
    @param source 调用来源（cli / agent）
    @param tool 工具名（日志用）
    @return 格式化文本
    """
    from llgraph.code_index.index_settings import resolve_index_settings

    logger = get_search_logger()
    skip_dirs = frozenset(resolve_index_settings(workspace).skip_dirs)
    ctx = WorkspaceContext(workspace, allow_write=False, extra_skip_dirs=skip_dirs)
    prefix = path_prefix.strip() if path_prefix else "."

    grep_list = _grep_hits(ctx, query, prefix, limit=GREP_TOP)
    vector_list, vector_err = _vector_hit_ids(
        workspace,
        query,
        prefix,
        VECTOR_TOP,
        source=source,
        tool=tool,
    )

    logger.info(
        "[vector] hybrid 汇总 source=%s tool=%s grep_hits=%d vector_hits=%d top_k=%d vector_err=%r",
        source,
        tool,
        len(grep_list),
        len(vector_list),
        top_k,
        vector_err or "",
    )

    scores: dict[str, float] = {}
    previews: dict[str, str] = {}

    for rank, (doc_id, snippet) in enumerate(grep_list, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + _rrf_score(rank)
        previews.setdefault(doc_id, snippet)

    for rank, (doc_id, snippet) in enumerate(vector_list, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + _rrf_score(rank)
        previews.setdefault(doc_id, snippet)

    if not scores:
        msg = "Hybrid 检索无结果。"
        if vector_err:
            msg += f" 向量路失败: {vector_err}"
        msg += " 请先 llgraph index -C . 或调整关键词。"
        return msg

    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    lines = [
        f"Hybrid 检索 Top{len(ranked)}（RRF k={RRF_K}, "
        f"grep={len(grep_list)}, vector={len(vector_list)}）:",
        "",
    ]
    for i, (doc_id, score) in enumerate(ranked, start=1):
        prev = previews.get(doc_id, "")
        lines.append(f"{i}. [{score:.4f}] {doc_id}")
        if prev:
            lines.append(f"   {prev}")
    if vector_err:
        lines.append(f"\n（向量路告警: {vector_err}）")
    lines.append("\n请对命中位置使用 read_file 精读。")
    return "\n".join(lines)
