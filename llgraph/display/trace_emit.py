"""工具内部子过程写入终端 trace（如 search_code_parallel 的检索参数）。"""

from __future__ import annotations

from typing import Any

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


def emit_invoke_prelude_step(
    session: Any,
    *,
    title: str,
    summary: str,
    elapsed: float,
    body_lines: list[str] | None = None,
    kind: str = "preprocess",
) -> int:
    """
    invoke 前预处理：仅写日志，不进入 Trace 步骤 / Web SSE。

    真正发生裁剪/压缩时仍由 emit_tool_prune_trace_step / emit_compress_trace_step 展示。

    @return 恒为 0（不占步骤号）
    """
    import logging

    logger = logging.getLogger("llgraph.trace.prelude")
    detail = ""
    if body_lines:
        detail = " | " + " · ".join(str(x) for x in body_lines[:4])
    logger.info(
        "invoke_prelude kind=%s title=%s summary=%s elapsed=%.3fs%s",
        kind,
        title,
        summary,
        elapsed,
        detail,
    )
    return 0


def emit_explore_trace_step(
    session: Any,
    *,
    title: str,
    summary: str,
    elapsed: float,
    sub_thread: str,
    body_lines: list[str] | None = None,
) -> int:
    """
    父会话 Trace：登记 explore 子 Agent 步骤（可展开拉子 Trace）。

    @param session TraceSession（父）
    @param title 如 Explore
    @param summary 折叠摘要
    @param elapsed 子 Agent 耗时秒
    @param sub_thread 子会话 thread_id
    @param body_lines 展开预览（摘要片段）
    @return 步骤编号；未展示时 0
    """
    from llgraph.display.trace_display import TraceStepRecord, _resolve_elapsed_kind

    if session is None or session.is_silent() or not session.shows_process():
        return 0
    sub = (sub_thread or "").strip()
    if not sub:
        return 0

    lines = list(body_lines or [])
    if not any(ln.startswith("sub_thread=") for ln in lines):
        lines = [f"sub_thread={sub}", *lines]

    printer = session.active_printer
    if printer is not None:
        step_id = printer._register_step(  # noqa: SLF001 — 共享登记路径
            "explore",
            title,
            elapsed,
            summary,
            body_lines=lines,
            elapsed_kind=_resolve_elapsed_kind("explore"),
            sub_thread=sub,
        )
        printer._print_step_summary(  # noqa: SLF001
            step_id,
            title,
            elapsed,
            summary,
            step_marker="◈",
            inline_preview=2,
        )
        return step_id

    step_id = len(getattr(session, "last_turn_steps", []) or []) + 1
    record = TraceStepRecord(
        step_id=step_id,
        kind="explore",
        title=title,
        elapsed=elapsed,
        summary=summary,
        body_lines=lines,
        elapsed_kind=_resolve_elapsed_kind("explore"),
        sub_thread=sub,
    )
    if hasattr(session, "last_turn_steps"):
        session.last_turn_steps.append(record)
    sink = session.trace_sink
    if sink is not None and hasattr(sink, "step_added"):
        sink.step_added(record)
    return step_id


def update_explore_trace_step(
    session: Any,
    step_id: int,
    *,
    summary: str,
    elapsed: float,
    body_lines: list[str] | None = None,
) -> bool:
    """
    更新已登记的 explore 步骤（同 step_id 回填摘要/耗时，避免跑完再插一条）。

    @param session TraceSession（父）
    @param step_id emit_explore_trace_step 返回值
    @param summary 折叠摘要
    @param elapsed 子 Agent 耗时秒
    @param body_lines 展开预览
    @return 是否更新成功
    """
    if session is None or step_id <= 0:
        return False

    record = None
    printer = session.active_printer
    if printer is not None:
        for step in getattr(printer, "_steps", []) or []:
            if int(getattr(step, "step_id", 0) or 0) == step_id:
                record = step
                break
    if record is None:
        for step in getattr(session, "last_turn_steps", []) or []:
            if int(getattr(step, "step_id", 0) or 0) == step_id:
                record = step
                break
    if record is None:
        return False

    record.summary = summary
    record.elapsed = float(elapsed)
    if body_lines is not None:
        lines = list(body_lines)
        sub = (getattr(record, "sub_thread", None) or "").strip()
        if sub and not any(ln.startswith("sub_thread=") for ln in lines):
            lines = [f"sub_thread={sub}", *lines]
        record.body_lines = lines

    sink = session.trace_sink
    if sink is not None and hasattr(sink, "step_added"):
        sink.step_added(record)
    return True


def emit_turn_epilogue_step(
    session: Any,
    *,
    title: str,
    summary: str,
    elapsed: float,
    body_lines: list[str] | None = None,
    kind: str = "preprocess",
) -> int:
    """
    轮次结束后的追加步骤：写入 last_turn_steps 并推送 SSE（不进下一轮 pending）。

    用于 active_printer 已清空、但本轮 turn_done 尚未发出时的追加步骤。

    @param session TraceSession
    @param title 步骤标题
    @param summary 折叠摘要
    @param elapsed 耗时秒
    @param body_lines 展开详情
    @param kind 步骤类型
    @return 步骤编号；未展示时 0
    """
    from llgraph.display.trace_display import TraceStepRecord, _resolve_elapsed_kind

    if session is None or session.is_silent() or not session.shows_process():
        return 0

    printer = session.active_printer
    if printer is not None:
        return printer.emit_preprocess_step(
            title,
            summary,
            body_lines or [],
            elapsed,
            kind=kind,
            inline_preview=2,
        )

    steps = list(getattr(session, "last_turn_steps", None) or [])
    step_id = len(steps) + 1
    record = TraceStepRecord(
        step_id=step_id,
        kind=kind,
        title=title,
        elapsed=elapsed,
        summary=summary,
        body_lines=body_lines or [],
        elapsed_kind=_resolve_elapsed_kind(kind),
    )
    steps.append(record)
    session.last_turn_steps = steps
    sink = session.trace_sink
    if sink is not None and hasattr(sink, "step_added"):
        sink.step_added(record)
    return step_id


def format_compress_skip_summary(
    agent: Any,
    *,
    thread_id: str,
    workspace: Any,
) -> str:
    """上下文未触发压缩时的 trace 摘要。"""
    from llgraph.context.context_compressor import ContextCompressor, estimate_tokens
    from llgraph.context.incremental_context import resolve_auto_compress_threshold

    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = agent.get_state(config)
        messages = list((state.values or {}).get("messages") or [])
    except Exception:
        messages = []
    compressor = ContextCompressor(workspace, thread_id)
    tokens = estimate_tokens(messages) if messages else 0
    threshold = resolve_auto_compress_threshold(compressor.settings)
    ratio_pct = int(compressor.settings.auto_compress_ratio * 100)
    return f"估算 ~{tokens} · 阈值 ~{threshold}（{ratio_pct}% 窗）· 未触发"


def format_sanitize_summary(report: Any) -> str:
    """canonical sanitize 步骤摘要。"""
    if report is None or not getattr(report, "changed", False):
        return "无变更"
    parts: list[str] = []
    if report.removed_orphan_tools:
        parts.append(f"移除 orphan tool ×{report.removed_orphan_tools}")
    if report.patched_tool_results:
        parts.append(f"补齐 tool 占位 ×{report.patched_tool_results}")
    if report.normalized_ai_messages:
        parts.append(f"规范化 AI ×{report.normalized_ai_messages}")
    if report.expanded_tool_rounds:
        parts.append(f"展开 tool 轮 ×{report.expanded_tool_rounds}")
    return " · ".join(parts) if parts else "已清理"


def emit_compress_trace_step(
    report: Any,
    *,
    thread_id: str,
    workspace: Any | None = None,
    session: Any | None = None,
    elapsed_sec: float | None = None,
) -> None:
    """
    在 trace 中注册「上下文压缩」步骤（invoke 前 / ReAct 中途）。

    @param report CompressReport
    @param thread_id 会话 thread
    @param workspace 工作区根（写 execution 日志）
    """
    from llgraph.context.context_compressor import CompressReport, format_compress_report
    from llgraph.display.trace_display import LAST_TRACE_SESSION

    if not isinstance(report, CompressReport) or not thread_id.strip():
        return

    session = session or LAST_TRACE_SESSION
    body = format_compress_report(report).splitlines()
    pct = int(report.saved_ratio * 100)
    summary = (
        f"token {report.before_tokens}→{report.after_tokens}（约释放 {pct}%）"
        f" · {report.trigger}"
    )
    if report.llm_sec > 0:
        summary += f" · LLM {_format_compress_elapsed(report.llm_sec)}"

    wall = elapsed_sec if elapsed_sec is not None else report.elapsed_sec
    if session is not None and session.shows_process():
        printer = session.active_printer
        if printer is not None:
            printer.emit_preprocess_step(
                "上下文压缩",
                summary,
                body,
                wall,
                kind="compress",
                inline_preview=4,
            )
        else:
            emit_invoke_prelude_step(
                session,
                title="上下文压缩",
                summary=summary,
                elapsed=wall,
                body_lines=body,
                kind="compress",
            )

    if workspace is not None:
        from llgraph.display.execution_log import log_compress_event

        log_compress_event(
            workspace,
            thread_id=thread_id,
            report=report,
            trigger=report.trigger or "auto",
        )


def _format_compress_elapsed(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def emit_tool_prune_trace_step(
    report: Any,
    *,
    thread_id: str,
    workspace: Any | None = None,
    session: Any | None = None,
    elapsed_sec: float | None = None,
) -> None:
    """
    在 trace 中注册「工具结果落盘」步骤（ReAct 步间增量写回）。

    @param report ToolPruneReport
    @param thread_id 会话 thread
    @param workspace 工作区根（预留 execution 日志）
    """
    from llgraph.context.incremental_context import ToolPruneReport, format_tool_prune_report
    from llgraph.display.trace_display import LAST_TRACE_SESSION

    if not isinstance(report, ToolPruneReport) or not thread_id.strip():
        return

    session = session or LAST_TRACE_SESSION
    body = format_tool_prune_report(report).splitlines()
    summary = (
        f"{report.pruned_count} 条→指针 · "
        f"token {report.before_tokens}→{report.after_tokens}"
    )
    if report.trigger:
        summary += f" · {report.trigger}"

    wall = elapsed_sec if elapsed_sec is not None else report.elapsed_sec
    if session is not None and session.shows_process():
        printer = session.active_printer
        if printer is not None:
            printer.emit_preprocess_step(
                "工具结果裁剪",
                summary,
                body,
                wall,
                kind="tool_prune",
                inline_preview=2,
            )
        else:
            emit_invoke_prelude_step(
                session,
                title="工具结果裁剪",
                summary=summary,
                elapsed=wall,
                body_lines=body,
                kind="tool_prune",
            )

    _ = workspace


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
