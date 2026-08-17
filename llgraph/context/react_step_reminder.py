"""ReAct 步间提醒：工具返回后、下一轮 LLM 前注入（仅 dispatch，不落盘）。

只提醒批量工具与往返预算；不按意图分流决策。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from llgraph.context.chat_history_repair import ai_message_tool_calls
from llgraph.context.investigate_harness import (
    count_tool_rounds_since_user,
    is_ephemeral_harness_human,
    last_real_user_text,
)

_NUDGE_MARKERS = (
    "<system-reminder>",
    "[系统·ReAct 步间提醒]",
    "[系统·探索预算]",
    "[系统·强制结案]",
    "[系统·防偏航收口]",
    "[系统] 你上一轮仅在 thinking",
)


def _is_ephemeral_nudge(msg: BaseMessage) -> bool:
    return is_ephemeral_harness_human(msg) or (
        isinstance(msg, HumanMessage)
        and any(
            str(getattr(msg, "content", "") or "").strip().startswith(m) for m in _NUDGE_MARKERS
        )
    )


def _tool_rounds_since_last_user(messages: list[BaseMessage]) -> int:
    return count_tool_rounds_since_user(messages)


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
        if not isinstance(msg, HumanMessage):
            continue
        body = str(getattr(msg, "content", "") or "").strip()
        if body.startswith("<system-reminder>") or body.startswith("[系统·ReAct"):
            return False
    return True


def react_step_reminder_content(
    messages: list[BaseMessage],
    *,
    workspace: Path | None = None,
) -> str:
    """
    批量工具与预算提醒（不按意图分流）。

    @param messages 规范化前或后的消息列表
    @param workspace 工作区根（预算上限）
    @return 提醒正文
    """
    from llgraph.core.react_limits import format_tool_round_budget_line

    excerpt = last_real_user_text(messages)
    if len(excerpt) > 180:
        excerpt = excerpt[:179] + "…"
    single = _last_ai_tool_call_count(messages) == 1
    if single:
        body = (
            "上一轮 assistant 仅 1 个 tool_call。若未完成，请并行多个独立工具，"
            "或一条多词 grep_files。\n"
        )
    else:
        body = "刚收到工具结果。未完成目标可批量补工具；证据够则写终答。\n"
    budget = format_tool_round_budget_line(messages, workspace=workspace)
    query_line = f'钉住 <user_query>："{excerpt}"\n' if excerpt else ""
    return f"<system-reminder>\n{query_line}{budget}\n{body}</system-reminder>"


def append_react_step_reminder_for_dispatch(
    messages: list[BaseMessage],
    *,
    workspace: Path | None = None,
) -> list[BaseMessage]:
    """
    工具返回后的 ephemeral 提醒（不写 checkpoint）；始终带工具往返剩余次数。

    @param messages 即将发往模型的消息
    @param workspace 工作区根
    @return 可能追加 HumanMessage 的副本
    """
    if should_inject_react_step_reminder(messages):
        content = react_step_reminder_content(messages, workspace=workspace)
        return [*messages, HumanMessage(content=content)]
    from llgraph.core.react_limits import format_tool_round_budget_line

    if not messages:
        return messages
    last = messages[-1]
    if isinstance(last, HumanMessage):
        body = str(getattr(last, "content", "") or "")
        if "工具往返预算：" in body:
            return messages
    budget = format_tool_round_budget_line(messages, workspace=workspace)
    return [
        *messages,
        HumanMessage(content=f"<system-reminder>\n{budget}\n</system-reminder>"),
    ]
