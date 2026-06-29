"""工具执行期上下文（供 tool 函数读取当前图消息，做 turn 内策略校验）。"""

from __future__ import annotations

from contextvars import ContextVar

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

_execution_messages: ContextVar[list[BaseMessage] | None] = ContextVar(
    "llgraph_tool_execution_messages",
    default=None,
)


def set_tool_execution_messages(messages: list[BaseMessage] | None) -> None:
    """
    工具节点 invoke 前写入当前 messages（含本轮 AI tool_calls 之前的历史）。

    @param messages 图状态 messages；None 表示清除
    """
    _execution_messages.set(messages)


def get_tool_execution_messages() -> list[BaseMessage]:
    """@return 当前工具执行上下文中的 messages；无则空列表"""
    raw = _execution_messages.get()
    return list(raw) if raw else []


def count_tool_results_since_user(
    messages: list[BaseMessage],
    tool_name: str,
) -> int:
    """
    统计自最近 user 消息以来某工具已成功执行的次数。

    @param messages 消息列表
    @param tool_name 工具名
    @return 计数
    """
    count = 0
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage) and str(getattr(msg, "name", "") or "") == tool_name:
            count += 1
    return count
