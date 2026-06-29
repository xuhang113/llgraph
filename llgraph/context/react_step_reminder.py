"""ReAct 步间批量工具提醒：工具返回后、下一轮 LLM 前注入（仅 dispatch，不落盘）。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from llgraph.context.chat_history_repair import ai_message_tool_calls

REACT_STEP_BATCH_REMINDER = (
    "[系统·ReAct 步间提醒] 你刚从工具返回。若还要继续调查："
    "仍需 ≥2 次 grep/read/glob/list → **同一条 assistant 消息内**一次发出多个 tool_call；"
    "多词检索 → 一条 `grep_files(pattern=\"a|b|c\", path=\".\")`；"
    "多 path → `read_files(paths=[...])`。"
    "证据已够 → **停 tool**，直接输出用户可见正文。"
)

REACT_STEP_SINGLE_TOOL_NUDGE = (
    "[系统·ReAct 步间提醒] 上一轮仅 **1 个** tool_call。"
    "若 thinking 里还有未执行的检索/读取目标，本轮必须合并为同消息多 tool_call "
    "或 `grep_files(pattern=\"a|b|c\")`；禁止再拆成下一轮单 grep/read。"
    "证据够则停 tool 直接答。"
)

_NUDGE_MARKERS = (
    "[系统·ReAct 步间提醒]",
    "[系统] 你上一轮仅在 thinking",
)


def _is_ephemeral_nudge(msg: BaseMessage) -> bool:
    if not isinstance(msg, HumanMessage):
        return False
    text = str(getattr(msg, "content", "") or "").strip()
    return any(text.startswith(m) for m in _NUDGE_MARKERS)


def _last_real_human_index(messages: list[BaseMessage]) -> int:
    """末条真实用户消息下标（跳过步间 nudge）。"""
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, HumanMessage) and not _is_ephemeral_nudge(msg):
            return idx
    return -1


def _tool_rounds_since_last_user(messages: list[BaseMessage]) -> int:
    """自末条用户消息以来，含 tool_calls 的 assistant 轮数。"""
    start = _last_real_human_index(messages)
    if start < 0:
        start = 0
    rounds = 0
    for msg in messages[start + 1 :]:
        if isinstance(msg, AIMessage) and ai_message_tool_calls(msg):
            rounds += 1
    return rounds


def _last_ai_tool_call_count(messages: list[BaseMessage]) -> int:
    """最近一条带 tool_calls 的 AIMessage 上的 call 数量。"""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            calls = ai_message_tool_calls(msg)
            if calls:
                return len(calls)
    return 0


def should_inject_react_step_reminder(messages: list[BaseMessage]) -> bool:
    """
    是否在发往 LLM 前注入步间批量提醒。

    条件：末条为 ToolMessage（刚从 tools 节点返回），且本轮已至少完成 1 次工具往返。
    """
    if not messages:
        return False
    if not isinstance(messages[-1], ToolMessage):
        return False
    if _tool_rounds_since_last_user(messages) < 1:
        return False
    for msg in reversed(messages[-3:]):
        if isinstance(msg, HumanMessage) and str(getattr(msg, "content", "") or "").startswith(
            "[系统·ReAct 步间提醒]"
        ):
            return False
    return True


def react_step_reminder_content(messages: list[BaseMessage]) -> str:
    """
    按上一轮 tool_call 数量选择提醒强度。

    @param messages 规范化前或后的消息列表
    @return 提醒正文
    """
    if _last_ai_tool_call_count(messages) == 1:
        return REACT_STEP_SINGLE_TOOL_NUDGE
    return REACT_STEP_BATCH_REMINDER


def append_react_step_reminder_for_dispatch(
    messages: list[BaseMessage],
) -> list[BaseMessage]:
    """
    工具返回后的 ephemeral 提醒（不写 checkpoint）。

    @param messages 即将发往模型的消息
    @return 可能追加 HumanMessage 的副本
    """
    if not should_inject_react_step_reminder(messages):
        return messages
    content = react_step_reminder_content(messages)
    return [*messages, HumanMessage(content=content)]
