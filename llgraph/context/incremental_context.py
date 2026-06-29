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


def prune_stale_tool_messages(
    messages: list[BaseMessage],
    workspace: Path,
    settings: ContextSettings,
) -> tuple[list[BaseMessage], int]:
    """
    保留最近 N 条 ToolMessage 全文，更早的超长 tool 输出替换为指针。

    @param messages 当前消息列表
    @param workspace 工作区根
    @param settings 上下文配置
    @return (新消息列表, 被裁剪条数)
    """
    if not settings.incremental_tool_prune:
        return messages, 0

    keep = settings.keep_recent_tool_messages
    mask_chars = settings.compress_tool_mask_max_chars
    tool_indices = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    if not tool_indices:
        return messages, 0

    keep_indices = set(tool_indices[-keep:]) if keep > 0 else set()
    new_messages: list[BaseMessage] = []
    pruned = 0
    for idx, msg in enumerate(messages):
        if not isinstance(msg, ToolMessage) or idx in keep_indices:
            new_messages.append(msg)
            continue
        masked = mask_tool_message_content(msg, workspace, max_chars=mask_chars)
        if masked.content != msg.content:
            pruned += 1
        new_messages.append(masked)
    return new_messages, pruned


def apply_incremental_tool_prune_to_agent_state(
    agent: Any,
    *,
    thread_id: str,
    workspace: Path,
) -> ToolPruneReport | None:
    """
    从 agent 状态读取消息、裁剪历史 tool 输出并写回。

    @param agent LangGraph agent
    @param thread_id 线程 ID
    @param workspace 工作区根
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
    try:
        agent.update_state(config, {"messages": new_messages})
        from llgraph.session.session_file_store import save_session_messages

        save_session_messages(workspace, thread_id, new_messages)
    except Exception:
        return None

    return ToolPruneReport(
        before_tokens=before_tokens,
        after_tokens=after_tokens,
        pruned_count=pruned_count,
    )


def prune_tool_messages_for_dispatch(
    messages: list[BaseMessage],
    workspace: Path,
    settings: ContextSettings,
) -> list[BaseMessage]:
    """
    发往模型前裁剪 tool 链（不修改 MemorySaver / 落盘）：仅最近 N 条 ToolMessage 保留全文。

    对齐 Cursor 动态上下文：grep/read 结果落盘或指针，模型按需 read_file 行段。

    @param messages 已 canonical / repair 后的消息
    @param workspace 工作区根
    @param settings 上下文配置
    @return 出站用消息列表
    """
    if not settings.dispatch_tool_chain_compress:
        return messages

    keep = settings.dispatch_keep_full_tool_messages
    mask_chars = settings.compress_tool_mask_max_chars
    tool_indices = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    if not tool_indices or len(tool_indices) <= keep:
        return messages

    keep_indices = set(tool_indices[-keep:])
    new_messages: list[BaseMessage] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, ToolMessage) or idx in keep_indices:
            new_messages.append(msg)
            continue
        from llgraph.context.context_spill import mask_tool_message_to_dispatch_pointer

        new_messages.append(mask_tool_message_to_dispatch_pointer(msg))
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
