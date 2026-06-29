"""出站上下文窗口：置顶 system + 最近 user 轮（固定 N 或按 token 自动扩展）。"""

from __future__ import annotations

from typing import Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from llgraph.context.chat_history_repair import ai_message_has_tool_calls
from llgraph.context.context_message_split import _segment_messages
from llgraph.context.context_settings import ContextSettings
from llgraph.context.conversation_anchor import is_conversation_anchor_message
from llgraph.context.message_canonical import is_session_archive_message
from llgraph.session.session_manifest import is_session_manifest_message


def _is_pinned_dispatch_message(msg: BaseMessage) -> bool:
    if not isinstance(msg, SystemMessage):
        return False
    return (
        is_session_manifest_message(msg)
        or is_conversation_anchor_message(msg)
        or is_session_archive_message(msg)
    )


def _split_pinned_business(
    messages: list[BaseMessage],
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    pinned = [m for m in messages if _is_pinned_dispatch_message(m)]
    business = [m for m in messages if m not in pinned]
    return pinned, business


def trim_messages_for_dispatch_window(
    messages: list[BaseMessage],
    *,
    keep_user_turns: int,
) -> list[BaseMessage]:
    """
    发往模型前裁剪业务消息：保留置顶 system + 最近 keep_user_turns 个 user 轮（含完整 tool 链）。

    落盘与 MemorySaver 状态不变，仅减少 API token。

    @param messages 已 canonical / repair 后的消息
    @param keep_user_turns 保留的 user 轮数；≤0 时不裁剪
    @return 裁剪后的消息列表
    """
    if keep_user_turns <= 0 or not messages:
        return messages

    pinned, business = _split_pinned_business(messages)
    if not business:
        return messages

    segments = _segment_messages(business)
    kept_segments: list[list[BaseMessage]] = []
    user_turns = 0
    for seg in reversed(segments):
        kept_segments.insert(0, seg)
        user_turns += sum(1 for m in seg if isinstance(m, HumanMessage))
        if user_turns >= keep_user_turns:
            break

    if not kept_segments:
        return messages

    keep_count = sum(len(s) for s in kept_segments)
    if keep_count >= len(business):
        return messages

    trimmed_business = business[-keep_count:]
    return [*pinned, *trimmed_business]


def trim_messages_for_dispatch_window_auto(
    messages: list[BaseMessage],
    *,
    settings: ContextSettings,
    estimate_tokens: Callable[[list[BaseMessage]], int],
) -> list[BaseMessage]:
    """
    按 token 预算自动扩展出站 user 轮：至少 min 轮，在预算内尽量多保留。

    @param messages 已 canonical / repair 后的消息
    @param settings 上下文配置
    @param estimate_tokens token 估算函数
    @return 裁剪后的消息列表
    """
    if not messages:
        return messages

    pinned, business = _split_pinned_business(messages)
    if not business:
        return messages

    token_budget = int(settings.max_tokens_estimate * settings.dispatch_window_token_ratio)
    min_turns = settings.dispatch_min_user_turns
    max_turns = settings.dispatch_max_user_turns

    segments = _segment_messages(business)
    kept_segments: list[list[BaseMessage]] = []
    user_turns = 0
    tool_rounds_kept = 0
    tokens_used = estimate_tokens(pinned)

    for seg in reversed(segments):
        seg_tokens = estimate_tokens(seg)
        has_user = any(isinstance(m, HumanMessage) for m in seg)
        is_tool_round = any(
            isinstance(m, AIMessage) and ai_message_has_tool_calls(m) for m in seg
        )
        if kept_segments:
            if user_turns >= max_turns:
                break
            over_budget = tokens_used + seg_tokens > token_budget
            if over_budget:
                if user_turns >= min_turns:
                    break
                if tool_rounds_kept >= settings.dispatch_min_tool_rounds:
                    break
        kept_segments.insert(0, seg)
        tokens_used += seg_tokens
        if has_user:
            user_turns += sum(1 for m in seg if isinstance(m, HumanMessage))
        if is_tool_round:
            tool_rounds_kept += 1

    if not kept_segments:
        return messages

    keep_count = sum(len(s) for s in kept_segments)
    if keep_count >= len(business):
        return messages

    return [*pinned, *business[-keep_count:]]


def apply_dispatch_window_trim(
    messages: list[BaseMessage],
    *,
    settings: ContextSettings,
    estimate_tokens: Callable[[list[BaseMessage]], int],
) -> list[BaseMessage]:
    """
    按配置应用出站窗口裁剪（固定 N 或 auto）。

    @param messages 消息列表
    @param settings 上下文配置
    @param estimate_tokens token 估算
    @return 裁剪后列表
    """
    if settings.dispatch_keep_user_turns > 0:
        return trim_messages_for_dispatch_window(
            messages,
            keep_user_turns=settings.dispatch_keep_user_turns,
        )
    from llgraph.context.context_settings import is_auto_compress_strategy

    if is_auto_compress_strategy(settings.compress_strategy):
        return trim_messages_for_dispatch_window_auto(
            messages,
            settings=settings,
            estimate_tokens=estimate_tokens,
        )
    return messages
