"""增量上下文：每轮前裁剪历史 ToolMessage，避免全量堆积（对齐 Cursor 按需注入思路）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage

from llgraph.context.context_compressor import estimate_tokens
from llgraph.context.context_settings import ContextSettings, is_auto_compress_strategy, resolve_context_settings
from llgraph.context.context_spill import mask_tool_message_content


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
