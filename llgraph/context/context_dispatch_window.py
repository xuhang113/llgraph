"""出站上下文窗口：仅保留最近若干 user 轮，远早历史靠 anchor + search_session_history（对齐 Cursor）。"""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from llgraph.context.context_message_split import _segment_messages
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

    pinned = [m for m in messages if _is_pinned_dispatch_message(m)]
    business = [m for m in messages if m not in pinned]
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
