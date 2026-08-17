"""Tool 链裁剪的 token 压力门控（接近满窗前不 mask 历史 tool）。"""

from __future__ import annotations

from langchain_core.messages import BaseMessage, ToolMessage

from llgraph.context.context_compressor import estimate_tokens
from llgraph.context.context_settings import ContextSettings


def compute_tool_prune_pressure(
    messages: list[BaseMessage],
    settings: ContextSettings,
) -> float:
    """
    计算 tool 裁剪压力：0=不裁剪，1=按配置全力裁剪。

    @param messages 当前消息列表
    @param settings 上下文配置
    @return 0.0～1.0
    """
    from llgraph.context.incremental_context import resolve_auto_compress_threshold

    if not messages:
        return 0.0
    tokens = estimate_tokens(messages)
    threshold = resolve_auto_compress_threshold(settings)
    if threshold <= 0:
        return 0.0
    start = int(threshold * settings.tool_prune_token_ratio)
    if tokens < start:
        return 0.0
    if tokens >= threshold:
        return 1.0
    return (tokens - start) / max(1, threshold - start)


def effective_tool_keep_count(
    total_tools: int,
    settings: ContextSettings,
    pressure: float,
    *,
    min_keep: int,
) -> int:
    """
    按压力插值保留全文 ToolMessage 条数。

    @param total_tools 当前 tool 条数
    @param settings 上下文配置
    @param pressure 裁剪压力
    @param min_keep 全力裁剪时至少保留条数
    @return 保留全文条数
    """
    if total_tools <= 0:
        return 0
    if pressure <= 0.0:
        return total_tools
    floor = max(1, min_keep)
    if pressure >= 1.0:
        return min(floor, total_tools)
    keep = int(total_tools - pressure * (total_tools - floor))
    return max(floor, min(total_tools, keep))


def count_tool_messages(messages: list[BaseMessage]) -> int:
    """@param messages 消息列表 @return ToolMessage 条数"""
    return sum(1 for m in messages if isinstance(m, ToolMessage))
