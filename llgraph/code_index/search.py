"""语义向量检索。"""

from __future__ import annotations

from pathlib import Path

from llgraph.code_index.search_path_filter import is_junk_search_path
from llgraph.code_index.vector_search import perform_vector_search
from llgraph.config.logging_settings import get_search_logger


def collect_vector_hits(
    workspace: Path,
    query: str,
    *,
    top_k: int,
    path_prefix: str = ".",
    source: str = "cli",
    tool: str = "search_code_semantic",
) -> tuple[list[tuple[str, str]], str]:
    """
    纯向量检索，返回结构化命中（供 parallel_search RRF 融合）。

    @param workspace 工作区根
    @param query 语义 query（通常为用户原句）
    @param top_k 条数
    @param path_prefix 路径前缀
    @param source 调用来源
    @param tool 工具名（日志用）
    @return ((doc_id, preview) 列表, 错误文案；无错为空串)
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
            "[vector] 检索失败 source=%s tool=%s error=%s",
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
        if is_junk_search_path(str(rel)):
            continue
        start = hit.get("start_line", 0)
        doc_id = f"{rel}:{start}"
        preview = hit.get("text_preview", "")
        out.append((doc_id, preview))
    return out, ""


def format_hit(hit: dict, rank: int) -> str:
    """
    格式化单条命中。

    @param hit LanceDB 记录
    @param rank 名次（1-based）
    @return 文本行
    """
    rel = hit.get("rel_path", "")
    start = hit.get("start_line", 0)
    end = hit.get("end_line", 0)
    preview = hit.get("text_preview", "")
    dist = hit.get("_distance", "")
    dist_s = f" dist={dist:.4f}" if isinstance(dist, (int, float)) else ""
    return f"{rank}. {rel}:{start}-{end}{dist_s}\n   {preview}"


def search_semantic(
    workspace: Path,
    query: str,
    *,
    top_k: int = 15,
    path_prefix: str = ".",
    source: str = "cli",
    tool: str = "search_code_semantic",
) -> str:
    """
    语义检索代码块。

    @param workspace 工作区根
    @param query 自然语言或关键词
    @param top_k 返回条数
    @param path_prefix 路径前缀过滤
    @param source 调用来源（cli / agent）
    @param tool 工具名（日志用）
    @return 格式化结果
    """
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
        return f"查询向量化失败: {exc}"

    if outcome.skipped:
        if outcome.skip_reason == "empty_index":
            from llgraph.code_index.store import get_index_status

            status = get_index_status(workspace)
            return (
                "代码索引为空。请先执行: llgraph index -C <工作区>\n"
                f"索引路径: {status.lance_path}"
            )
        return "未找到语义相关代码块。"

    hits = outcome.hits
    if not hits:
        return "未找到语义相关代码块。"

    lines = [f"语义检索 Top{len(hits)}（索引 {outcome.chunk_count} chunks）:", ""]
    rank = 0
    for hit in hits:
        rel = str(hit.get("rel_path", ""))
        if is_junk_search_path(rel):
            continue
        rank += 1
        lines.append(format_hit(hit, rank))
        if rank >= top_k:
            break
    if rank == 0:
        return "未找到语义相关代码块（结果均为索引噪声，可 llgraph index 重建）。"
    return "\n".join(lines)
