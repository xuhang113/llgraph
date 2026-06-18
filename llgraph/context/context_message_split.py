"""压缩前消息切分（Tier 1）：按 token 预算保留最近段，且保持 tool 调用链完整。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from llgraph.context.chat_history_repair import ai_message_has_tool_calls


def _segment_messages(messages: list[BaseMessage]) -> list[list[BaseMessage]]:
    """
    将消息切成段：含 tool_calls 的 assistant 与其后 ToolMessage 同属一段。

    @param messages 消息列表
    @return 段列表
    """
    segments: list[list[BaseMessage]] = []
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]
        if isinstance(msg, ToolMessage):
            tools: list[BaseMessage] = []
            while i < n and isinstance(messages[i], ToolMessage):
                tools.append(messages[i])
                i += 1
            if segments and isinstance(segments[-1][0], AIMessage) and ai_message_has_tool_calls(
                segments[-1][0]
            ):
                segments[-1].extend(tools)
            continue
        if isinstance(msg, AIMessage) and ai_message_has_tool_calls(msg):
            seg = [msg]
            i += 1
            while i < n and isinstance(messages[i], ToolMessage):
                seg.append(messages[i])
                i += 1
            segments.append(seg)
            continue
        segments.append([msg])
        i += 1
    return segments


def split_messages_for_compress(
    messages: list[BaseMessage],
    *,
    token_budget: int,
    min_user_turns: int,
    estimate_tokens,
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """
    按 token 预算从尾部保留消息段；不足 min_user_turns 时多保留。

    @param messages 待切分消息（已去掉置顶 manifest/anchor）
    @param token_budget 保留段 token 上限
    @param min_user_turns 至少保留的 user 轮数（兜底）
    @param estimate_tokens token 估算函数
    @return (to_compress, to_keep)
    """
    if not messages:
        return [], []

    segments = _segment_messages(messages)
    kept_segments: list[list[BaseMessage]] = []
    tokens_used = 0
    user_turns = 0

    for seg in reversed(segments):
        seg_tokens = estimate_tokens(seg)
        has_user = any(isinstance(m, HumanMessage) for m in seg)
        if kept_segments:
            if tokens_used + seg_tokens > token_budget and user_turns >= min_user_turns:
                break
        kept_segments.insert(0, seg)
        tokens_used += seg_tokens
        if has_user:
            user_turns += sum(1 for m in seg if isinstance(m, HumanMessage))

    if not kept_segments:
        return list(messages), []

    keep_count = sum(len(s) for s in kept_segments)
    if keep_count >= len(messages):
        return [], list(messages)

    split_index = len(messages) - keep_count
    to_compress = list(messages[:split_index])
    to_keep = list(messages[split_index:])
    return to_compress, to_keep


def _last_human_index(messages: list[BaseMessage]) -> int:
    """
    最后一条 HumanMessage 下标；无则 -1。

    @param messages 消息列表
    @return 下标
    """
    last = -1
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            last = i
    return last


def split_messages_cursor_style(
    messages: list[BaseMessage],
    *,
    preserve_current_turn: bool,
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """
    Cursor 对齐切分：摘要由 LLM 生成，不做固定轮数 tail 裁剪。

    - preserve_current_turn=False（用户新消息前）：远早对话全部进摘要，换窗后仅 manifest+anchor
    - preserve_current_turn=True（ReAct 中途）：保留当前 user 轮（含完整 tool 链），更早进摘要

    @param messages 待切分消息（已去掉置顶 manifest/anchor）
    @param preserve_current_turn 是否保留当前 user 轮
    @return (to_compress, to_keep)
    """
    if not messages:
        return [], []

    if not preserve_current_turn:
        return list(messages), []

    last_human = _last_human_index(messages)
    if last_human <= 0:
        return [], list(messages)

    to_compress = list(messages[:last_human])
    to_keep = list(messages[last_human:])
    if not to_compress:
        return [], list(messages)
    return to_compress, to_keep


def split_messages_for_compress_strategy(
    messages: list[BaseMessage],
    *,
    strategy: str,
    preserve_current_turn: bool,
    token_budget: int,
    min_user_turns: int,
    estimate_tokens,
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """
    按压缩策略切分消息。

    @param messages 待切分消息
    @param strategy cursor | legacy
    @param preserve_current_turn cursor 策略下是否保留当前 user 轮
    @param token_budget legacy 保留段 token 上限
    @param min_user_turns legacy 至少保留 user 轮数
    @param estimate_tokens token 估算函数
    @return (to_compress, to_keep)
    """
    if strategy in ("cursor", "auto"):
        return split_messages_cursor_style(
            messages,
            preserve_current_turn=preserve_current_turn,
        )
    return split_messages_for_compress(
        messages,
        token_budget=token_budget,
        min_user_turns=min_user_turns,
        estimate_tokens=estimate_tokens,
    )
