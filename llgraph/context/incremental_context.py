"""增量上下文：每轮前裁剪历史 ToolMessage，避免全量堆积（对齐 Cursor 按需注入思路）。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage

from llgraph.context.context_compressor import estimate_tokens
from llgraph.context.context_settings import ContextSettings, is_auto_compress_strategy, resolve_context_settings
from llgraph.context.context_spill import mask_tool_message_content
from llgraph.context.tool_prune_pressure import (
    compute_tool_prune_pressure,
    effective_tool_keep_count,
)
from llgraph.context.read_segment_dedupe import (
    extract_read_segments,
    format_superseded_read_pointer,
    read_message_fully_superseded,
)

_READ_TOOL_NAMES = frozenset({"read_file", "read_files"})
_READ_PATH_HDR = re.compile(
    r"^---\s+(.+?)\s+\(行\s+\d+",
    re.MULTILINE,
)
_ARCHIVED_MARKERS = (
    "[历史",
    "[工具结果已落盘",
    "已 superseded",
    "已替换",
)


def _tool_content_is_archived(content: str) -> bool:
    return any(marker in content for marker in _ARCHIVED_MARKERS)


def _extract_read_source_paths(content: str) -> list[str]:
    """@param content read 工具输出 @return 源文件相对路径列表"""
    paths: list[str] = []
    seen: set[str] = set()
    for match in _READ_PATH_HDR.finditer(content):
        rel = match.group(1).strip()
        if rel and rel not in seen:
            seen.add(rel)
            paths.append(rel)
    return paths


@dataclass
class ToolPruneReport:
    """历史工具输出裁剪报告。"""

    before_tokens: int
    after_tokens: int
    pruned_count: int
    elapsed_sec: float = 0.0
    trigger: str = "invoke"

    @property
    def saved_tokens(self) -> int:
        return max(0, self.before_tokens - self.after_tokens)


def resolve_auto_compress_threshold(settings: ContextSettings) -> int:
    """
    自动压缩触发 token 上限（比例阈值与绝对上限取较小值）。

    @param settings 上下文配置
    @return 触发压缩的估算 token 数
    """
    ratio_threshold = int(settings.max_tokens_estimate * settings.auto_compress_ratio)
    if settings.compress_trigger_max_tokens is not None:
        return min(ratio_threshold, settings.compress_trigger_max_tokens)
    # auto 策略默认仅按比例触发，接近满窗再摘要
    if is_auto_compress_strategy(settings.compress_strategy):
        return ratio_threshold
    # legacy：大窗口模型默认加绝对上限
    if settings.max_tokens_estimate > 128_000:
        return min(ratio_threshold, 64_000)
    return ratio_threshold


def _protected_cited_indices(
    messages: list[BaseMessage],
    settings: ContextSettings,
    pressure: float,
    *,
    already_kept: set[int],
) -> set[int]:
    """
    下游引用信号：非满窗压力时，把被后续 AI 引用过的 ToolMessage 额外纳入保护。

    @param messages 消息列表
    @param settings 上下文配置
    @param pressure 当前裁剪压力
    @param already_kept 已按 recency 保留的下标
    @return 额外保护的下标集合（已去掉 already_kept、按上限截断）
    """
    if not settings.protect_cited_tool_messages:
        return set()
    if settings.max_protected_cited_tool_messages <= 0:
        return set()
    # 满窗压力（>=1）时不再保护，优先回收 token
    if pressure >= 1.0:
        return set()
    from llgraph.context.context_citation import cited_tool_indices

    cited = cited_tool_indices(messages) - already_kept
    if not cited:
        return set()
    # 只保护最近的 N 条被引用项，避免无界膨胀
    keep_recent_cited = sorted(cited)[-settings.max_protected_cited_tool_messages :]
    return set(keep_recent_cited)


def _prune_or_mask_tool_message(
    msg: ToolMessage,
    workspace: Path,
    settings: ContextSettings,
    *,
    cited_pairs: list[tuple[str, int]] | None = None,
) -> ToolMessage:
    """裁剪单条 ToolMessage；被引用行可选附「引用区预览」。"""
    tool_name = str(getattr(msg, "name", "") or "")
    mask_chars = (
        settings.read_tool_mask_max_chars
        if tool_name in _READ_TOOL_NAMES
        else settings.compress_tool_mask_max_chars
    )
    original = msg.content if isinstance(msg.content, str) else str(msg.content or "")
    masked = mask_tool_message_content(msg, workspace, max_chars=mask_chars)
    if masked.content == msg.content or not cited_pairs:
        return masked
    from llgraph.context.context_citation import build_cited_line_preview

    preview = build_cited_line_preview(original, cited_pairs)
    if not preview:
        return masked
    body = masked.content if isinstance(masked.content, str) else str(masked.content or "")
    enriched = f"{body}\n--- 被引用行预览（后续结论引用过） ---\n{preview}\n--- 预览结束 ---"
    return ToolMessage(
        content=enriched,
        tool_call_id=masked.tool_call_id,
        name=getattr(masked, "name", None),
    )


def prune_stale_tool_messages(
    messages: list[BaseMessage],
    workspace: Path,
    settings: ContextSettings,
) -> tuple[list[BaseMessage], int]:
    """
    保留最近 N 条 ToolMessage 全文，更早的超长 tool 输出替换为指针。

    被后续 AI 引用过的 ToolMessage 优先保留（下游引用信号，非满窗压力时生效）；
    若仍被裁剪，指针里附「被引用行 ± 上下文」预览。

    @param messages 当前消息列表
    @param workspace 工作区根
    @param settings 上下文配置
    @return (新消息列表, 被裁剪条数)
    """
    if not settings.incremental_tool_prune:
        return messages, 0

    tool_indices = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    if not tool_indices:
        return messages, 0

    pressure = compute_tool_prune_pressure(messages, settings)
    keep = effective_tool_keep_count(
        len(tool_indices),
        settings,
        pressure,
        min_keep=settings.keep_recent_tool_messages,
    )
    keep_indices = set(tool_indices[-keep:]) if keep > 0 else set()
    keep_indices |= _protected_cited_indices(
        messages, settings, pressure, already_kept=keep_indices
    )
    if keep_indices.issuperset(tool_indices):
        return messages, 0

    from llgraph.context.context_citation import cited_line_pairs_for_tool

    new_messages: list[BaseMessage] = []
    pruned = 0
    for idx, msg in enumerate(messages):
        if not isinstance(msg, ToolMessage) or idx in keep_indices:
            new_messages.append(msg)
            continue
        cited_pairs = cited_line_pairs_for_tool(messages, idx) if settings.protect_cited_tool_messages else None
        masked = _prune_or_mask_tool_message(
            msg, workspace, settings, cited_pairs=cited_pairs
        )
        if masked.content != msg.content:
            pruned += 1
        new_messages.append(masked)
    return new_messages, pruned


def apply_incremental_tool_prune_to_agent_state(
    agent: Any,
    *,
    thread_id: str,
    workspace: Path,
    trigger: str = "invoke",
    persist: bool | None = None,
) -> ToolPruneReport | None:
    """
    从 agent 状态读取消息、裁剪历史 tool 输出并写回。

    @param agent LangGraph agent
    @param thread_id 线程 ID
    @param workspace 工作区根
    @param trigger invoke | react（react 默认仅写 checkpoint，避免步间全量 jsonl 阻塞）
    @param persist 是否落盘 messages.jsonl；None 时 react 为 False、invoke 为 True
    @return 裁剪报告；无变化或未启用时返回 None
    """
    settings = resolve_context_settings(workspace)
    if not settings.incremental_tool_prune:
        return None

    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = agent.get_state(config)
    except Exception:
        return None
    messages = list((state.values or {}).get("messages") or [])
    if not messages:
        return None

    before_tokens = estimate_tokens(messages)
    new_messages, pruned_count = prune_stale_tool_messages(messages, workspace, settings)
    if pruned_count == 0:
        return None

    after_tokens = estimate_tokens(new_messages)
    write_disk = persist if persist is not None else trigger != "react"
    try:
        agent.update_state(config, {"messages": new_messages})
        if write_disk:
            from llgraph.session.session_file_store import save_agent_session_messages

            save_agent_session_messages(workspace, thread_id, new_messages, sync_pool=True)
    except Exception as exc:
        from llgraph.session.session_run_log import log_react_phase

        log_react_phase(
            workspace,
            thread_id,
            phase="tool_prune_persist_error",
            detail={"pruned_count": pruned_count, "trigger": trigger},
            error=exc,
        )
        return None

    return ToolPruneReport(
        before_tokens=before_tokens,
        after_tokens=after_tokens,
        pruned_count=pruned_count,
        trigger=trigger,
    )


def maybe_prune_tools_during_react(
    agent: Any,
    *,
    thread_id: str,
    workspace: Path,
) -> ToolPruneReport | None:
    """
    ReAct 每步 tools 节点结束后：将较早 ToolMessage 掩码写回 checkpoint。

    与 dispatch 出站裁剪互补：落盘会话不再只增不减，细节靠 search_session_history 等检索。

    @param agent LangGraph agent
    @param thread_id 线程 ID
    @param workspace 工作区根
    @return 裁剪报告；无变化或未启用时返回 None
    """
    import time

    settings = resolve_context_settings(workspace)
    if not settings.incremental_tool_prune:
        return None

    started = time.perf_counter()
    report = apply_incremental_tool_prune_to_agent_state(
        agent,
        thread_id=thread_id,
        workspace=workspace,
        trigger="react",
    )
    if report is None:
        return None
    report.elapsed_sec = time.perf_counter() - started
    return report


def pinned_write_success_indices(
    messages: list[BaseMessage],
    *,
    cap: int,
) -> set[int]:
    """
    每个已写路径保留最新一次成功写入（含写入后快照）。

    旧 read 出站作废后，快照是继续 search_replace 的原文；不能按 recency 丢掉。

    @param messages 出站消息
    @param cap 最多钉住的不同路径数
    @return 消息下标集合
    """
    if cap <= 0:
        return set()
    from llgraph.context.stale_read_after_write import collect_write_success_paths

    latest_by_path: dict[str, int] = {}
    for idx, path in collect_write_success_paths(messages):
        key = path or f"#{idx}"
        latest_by_path[key] = idx
    ranked = sorted(latest_by_path.values(), reverse=True)
    return set(ranked[:cap])


def dispatch_keep_tool_indices(
    messages: list[BaseMessage],
    settings: ContextSettings,
    *,
    thread_id: str | None = None,
) -> set[int]:
    """
    出站应保留全文的 ToolMessage 下标。

    压缩由「纪元水位」决定而非 recency 滑窗：全文重结果在高水位以下时一条都不新压，
    出站字节与上一步逐字节相同，prompt cache 整段命中；跨过高水位才一次压到低水位。
    轻量指针/拦截文案不占预算；写入快照按路径钉住最新一份；被引用项在尚未压缩前延后压缩。

    @param messages 出站消息
    @param settings 上下文配置
    @param thread_id 会话线程（记压缩水位，保证不回退）；None 时为纯函数模式
    @return 保留下标
    """
    from llgraph.context.dispatch_compaction import (
        plan_dispatch_compaction,
        tool_content_is_compact,
    )

    tool_indices = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    if not tool_indices:
        return set()

    light = {
        idx
        for idx in tool_indices
        if tool_content_is_compact(
            messages[idx].content
            if isinstance(messages[idx].content, str)
            else str(messages[idx].content or "")
        )
    }
    keep_n = max(1, settings.dispatch_keep_full_tool_messages)
    pinned = pinned_write_success_indices(messages, cap=max(4, keep_n))
    protected = _protected_cited_indices(
        messages,
        settings,
        pressure=0.0,
        already_kept=light | pinned,
    )
    plan = plan_dispatch_compaction(
        messages,
        settings,
        thread_id=thread_id,
        pinned=pinned,
        protected=protected,
    )
    return set(tool_indices) - set(plan.compact_indices)


def prune_tool_messages_for_dispatch(
    messages: list[BaseMessage],
    workspace: Path,
    settings: ContextSettings,
    *,
    thread_id: str | None = None,
) -> list[BaseMessage]:
    """
    发往模型前裁剪 tool 链（不修改 MemorySaver / 落盘）。

    对标 Claude Code / Codex CLI：出站按「压缩纪元」而非每步滑窗——全文重结果在
    高水位以下一条都不新压，跨过高水位才一次压到低水位。这样工具循环里除压缩那一步，
    出站前缀逐字节不变，prompt cache 整段命中（TTFT 与 input 计费都省）。
    模型若再调相同 read/grep，由 tool_loop_guard 短路径返回摘录。

    @param messages 已 canonical / repair 后的消息
    @param workspace 工作区根
    @param settings 上下文配置
    @param thread_id 会话线程（记压缩水位，保证压缩单调不回退）
    @return 出站用消息列表
    """
    if not settings.dispatch_tool_chain_compress:
        return messages

    tool_indices = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    if not tool_indices:
        return messages

    keep_indices = dispatch_keep_tool_indices(messages, settings, thread_id=thread_id)
    if keep_indices.issuperset(tool_indices):
        return messages

    from llgraph.context.context_citation import cited_line_pairs_for_tool
    from llgraph.context.context_spill import mask_tool_message_to_dispatch_pointer

    new_messages: list[BaseMessage] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, ToolMessage) or idx in keep_indices:
            new_messages.append(msg)
            continue
        pointer = mask_tool_message_to_dispatch_pointer(msg)
        if settings.protect_cited_tool_messages and pointer.content != msg.content:
            original = msg.content if isinstance(msg.content, str) else str(msg.content or "")
            pairs = cited_line_pairs_for_tool(messages, idx)
            if pairs:
                from llgraph.context.context_citation import build_cited_line_preview

                preview = build_cited_line_preview(original, pairs)
                if preview:
                    body = pointer.content if isinstance(pointer.content, str) else str(pointer.content or "")
                    pointer = ToolMessage(
                        content=f"{body}\n--- 被引用行预览（后续结论引用过） ---\n{preview}\n--- 预览结束 ---",
                        tool_call_id=pointer.tool_call_id,
                        name=getattr(pointer, "name", None),
                    )
        new_messages.append(pointer)
    return new_messages


def dedupe_read_tool_messages_for_dispatch(
    messages: list[BaseMessage],
    settings: ContextSettings,
) -> list[BaseMessage]:
    """
    出站去重：同文件多次 read 时，仅当行段重叠 ≥50% 才用较新 read 替换较旧。

    互补行段（如 1–120 与 460–480）均保留，避免中间逻辑被误删。

    @param messages 已 prune 后的消息
    @param settings 上下文配置
    @return 去重后的消息列表
    """
    if not settings.dispatch_dedupe_read_paths:
        return messages

    segments_by_idx: dict[int, list[tuple[str, int, int]]] = {}
    for idx, msg in enumerate(messages):
        if not isinstance(msg, ToolMessage):
            continue
        name = str(getattr(msg, "name", "") or "")
        if name not in _READ_TOOL_NAMES:
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if _tool_content_is_archived(content):
            continue
        segs = extract_read_segments(content)
        if segs:
            segments_by_idx[idx] = segs

    if not segments_by_idx:
        return messages

    new_messages: list[BaseMessage] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, ToolMessage):
            new_messages.append(msg)
            continue
        name = str(getattr(msg, "name", "") or "")
        if name not in _READ_TOOL_NAMES:
            new_messages.append(msg)
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if _tool_content_is_archived(content):
            new_messages.append(msg)
            continue
        segments = segments_by_idx.get(idx)
        if not segments:
            new_messages.append(msg)
            continue
        later_segments: list[tuple[str, int, int]] = []
        for later_idx, later_segs in segments_by_idx.items():
            if later_idx <= idx:
                continue
            later_segments.extend(later_segs)
        if not read_message_fully_superseded(idx, segments, later_segments):
            new_messages.append(msg)
            continue
        short = format_superseded_read_pointer(segments)
        new_messages.append(
            ToolMessage(
                content=short,
                tool_call_id=msg.tool_call_id,
                name=getattr(msg, "name", None),
            )
        )
    return new_messages


def format_tool_prune_report(report: ToolPruneReport) -> str:
    """
    格式化裁剪报告。

    @param report 裁剪报告
    @return 单行摘要
    """
    return (
        f"历史工具输出已裁剪: {report.pruned_count} 条→指针, "
        f"估算 token {report.before_tokens}→{report.after_tokens} "
        f"(约释放 {report.saved_tokens})"
    )
