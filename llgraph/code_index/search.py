"""语义向量检索。"""

from __future__ import annotations

from pathlib import Path

from llgraph.code_index.vector_search import perform_vector_search


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
    for i, hit in enumerate(hits, start=1):
        lines.append(format_hit(hit, i))
    return "\n".join(lines)
