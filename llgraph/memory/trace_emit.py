"""长期记忆 trace 步骤。"""

from __future__ import annotations

from typing import Any

from llgraph.memory.recall import MemoryRecallReport
from llgraph.memory.write import MemoryWriteReport


def emit_memory_recall_trace_step(
    report: MemoryRecallReport,
    *,
    session: Any | None = None,
) -> None:
    """注册「长期记忆 · 召回」步骤。"""
    from llgraph.display.trace_display import LAST_TRACE_SESSION
    from llgraph.display.trace_emit import emit_invoke_prelude_step

    if not report.hits:
        return
    session = session or LAST_TRACE_SESSION
    if session is None or session.is_silent() or not session.shows_process():
        return
    summary = (
        f"命中 {len(report.hits)} 条 · score≥阈值 · 约 {report.inject_chars} 字符"
    )
    body = [
        f"query: {report.query}",
        f"top_k: {report.top_k}",
        "hits:",
    ]
    for i, hit in enumerate(report.hits, start=1):
        mid = hit.memory_id[:8] + "…" if len(hit.memory_id) > 8 else hit.memory_id
        sim = getattr(hit, "similarity", None)
        sim_s = f" sim={sim:.2f}" if isinstance(sim, (int, float)) else ""
        body.append(
            f"  {i}. [{hit.kind}] score={hit.score:.2f}{sim_s} hit_count={hit.hit_count} id={mid} "
            f"{hit.content[:80]}{'…' if len(hit.content) > 80 else ''}"
        )
    if report.filtered_below_min:
        body.append(f"filtered: {report.filtered_below_min}（低于相似度/融合分门槛）")
    body.append(f"elapsed: {report.elapsed_ms:.0f}ms")
    emit_invoke_prelude_step(
        session,
        title="长期记忆 · 召回",
        summary=summary,
        elapsed=report.elapsed_ms / 1000.0,
        body_lines=body,
        kind="memory_recall",
    )


def emit_memory_write_trace_step(
    report: MemoryWriteReport,
    *,
    session: Any | None = None,
    elapsed_sec: float = 0.0,
) -> None:
    """注册「长期记忆 · 写入」步骤。"""
    from llgraph.display.trace_display import LAST_TRACE_SESSION
    from llgraph.display.trace_emit import emit_invoke_prelude_step, emit_turn_epilogue_step

    if report.action == "skip":
        return
    session = session or LAST_TRACE_SESSION
    if session is None or session.is_silent() or not session.shows_process():
        return
    summary = f"{report.action} · [{report.kind}] {report.memory_id[:8]}…"
    body = [
        f"action: {report.action}",
        f"kind: {report.kind}",
        f"memory_id: {report.memory_id}",
        f"reason: {report.reason}",
        f"content_preview: {report.content_preview}",
    ]
    if report.replaced_ids:
        body.append(f"replaced_ids: {report.replaced_ids}")
    emit_kw = dict(
        title="长期记忆 · 写入",
        summary=summary,
        elapsed=elapsed_sec,
        body_lines=body,
        kind="memory_write",
    )
    # 热路径 manage_memory：有 active_printer 时即时展示，否则挂到 last_turn_steps。
    if getattr(session, "active_printer", None) is not None:
        emit_invoke_prelude_step(session, **emit_kw)
    else:
        emit_turn_epilogue_step(session, **emit_kw)


def emit_memory_consolidate_trace_step(
    *,
    merged: int,
    replaced: int,
    pruned_ttl: int,
    pruned_cap: int,
    elapsed_sec: float,
    session: Any | None = None,
) -> None:
    """注册「长期记忆 · 整理」步骤。"""
    from llgraph.display.trace_display import LAST_TRACE_SESSION
    from llgraph.display.trace_emit import emit_invoke_prelude_step

    if merged + replaced + pruned_ttl + pruned_cap <= 0:
        return
    session = session or LAST_TRACE_SESSION
    if session is None or session.is_silent() or not session.shows_process():
        return
    summary = f"merged: {merged} · replaced: {replaced} · pruned_ttl: {pruned_ttl} · pruned_cap: {pruned_cap}"
    emit_invoke_prelude_step(
        session,
        title="长期记忆 · 整理",
        summary=summary,
        elapsed=elapsed_sec,
        body_lines=[summary, f"elapsed: {elapsed_sec:.2f}s"],
        kind="memory_consolidate",
    )
