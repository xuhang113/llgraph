"""工具内部子过程写入终端 trace（如 search_code_parallel 的检索参数）。"""

from __future__ import annotations

from llgraph.code_index.search_params import ParallelSearchParams

_SEARCH_PARAMS_INLINE_PREVIEW = 8


def format_search_params_trace_body(params: ParallelSearchParams) -> list[str]:
    """
    格式化检索参数详情（供 trace 步骤 body）。

    @param params 启发式检索参数
    @return 多行文本
    """
    lines: list[str] = [
        "source: 主 Agent query 启发式拆分",
        f"semantic_query: {params.semantic_query}",
        "",
    ]
    if params.literal_grep_patterns:
        lines.append(
            f"literal_grep_patterns（共 {len(params.literal_grep_patterns)} 条）:"
        )
        for idx, pat in enumerate(params.literal_grep_patterns, start=1):
            lines.append(f"  {idx}. {pat}")
    else:
        lines.append("literal_grep_patterns: (无，仅走向量)")
    lines.append("")
    lines.append("上下游扩展: 由主 Agent 后续 grep_files / read_files 负责")
    return lines


def format_search_params_trace_summary(params: ParallelSearchParams) -> str:
    """
    检索参数步骤折叠行摘要。

    @param params 启发式检索参数
    @return 单行摘要
    """
    n_lit = len(params.literal_grep_patterns)
    sem = (params.semantic_query or "").strip()
    sem_preview = sem[:36] + ("…" if len(sem) > 36 else "")
    parts = [f"literal×{n_lit}"]
    if sem_preview:
        parts.append(f"semantic={sem_preview!r}")
    return " · ".join(parts)


def emit_parallel_search_params_trace(
    params: ParallelSearchParams,
    *,
    elapsed: float,
    parent_tool: str = "search_code_parallel",
    user_query: str = "",
) -> None:
    """
    在工具执行 trace 中注册「检索参数」前置步骤（仅交互会话 steps/all 模式）。

    @param params 启发式检索参数
    @param elapsed 拆参耗时（秒）
    @param parent_tool 所属工具名
    @param user_query 主 Agent 传入的原始 query
    """
    from llgraph.display.trace_display import LAST_TRACE_SESSION

    session = LAST_TRACE_SESSION
    if session is None or not session.shows_process():
        return
    printer = session.active_printer
    if printer is None:
        return

    body = format_search_params_trace_body(params)
    if user_query.strip():
        body.insert(0, f"agent_query: {user_query.strip()}")
        body.insert(1, "")

    summary = format_search_params_trace_summary(params)

    printer.emit_preprocess_step(
        f"检索参数 · {parent_tool}",
        summary,
        body,
        elapsed,
        kind="search_params",
        inline_preview=_SEARCH_PARAMS_INLINE_PREVIEW,
    )
