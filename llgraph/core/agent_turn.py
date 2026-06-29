"""ReAct turn 完成判定：Think 与面向用户正文分离（对齐 Cursor think step 语义）。"""

from __future__ import annotations

from typing import Callable, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END

from llgraph.context.chat_history_repair import ai_message_has_tool_calls, ai_message_tool_calls
from llgraph.core.llm_response import llm_content_text

FALLBACK_INCOMPLETE_TURN = (
    "模型未能在本轮输出可见正文（Thinking 不能代替回复）。"
    "请重试或简化问题。"
)

THINK_CONTINUE_NUDGE = (
    "[系统] 你上一轮仅在 thinking/reasoning 中推理，未输出用户可见的正文 text。"
    "请基于已有思考，在本轮助手消息的 **正文 text** 中给出完整答复"
    "（Plan 模式须在正文输出 JSON 代码块）。勿再次 thinking-only 结束。"
)

RouteAfterAgent = Literal["tools", "think_nudge", "turn_fallback", "__end__"]


def ai_message_has_visible_text(msg: AIMessage) -> bool:
    """
    是否含用户可见正文（仅 type:text，不含 thinking；不含【规划】行）。

    @param msg assistant 消息
    @return 是否有可见 text
    """
    from llgraph.context.message_normalize import format_agent_chat_display_text

    raw = llm_content_text(msg.content, fallback_thinking=False)
    return bool(format_agent_chat_display_text(raw).strip())


def last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    """
    取消息列表中最后一条 AIMessage。

    @param messages 消息列表
    @return 最后一条 AI 或 None
    """
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    return None


def pending_tool_calls(
    messages: list[BaseMessage],
    *,
    last_ai: AIMessage | None = None,
) -> list[dict]:
    """
    最后一条 AIMessage 上尚未有 ToolMessage 应答的 tool_calls。

    仅统计 **该条 AI 之后** 的 ToolMessage，避免 Kimi 等复用
    ``functions.foo:1`` 时误把新一轮 call 当成历史已完成。

    @param messages 消息列表
    @param last_ai 可选，已知的最后 AI 消息
    @return pending calls
    """
    ai = last_ai if last_ai is not None else last_ai_message(messages)
    if ai is None:
        return []

    last_ai_idx = -1
    for idx, msg in enumerate(messages):
        if msg is ai:
            last_ai_idx = idx
    if last_ai_idx < 0:
        for idx in range(len(messages) - 1, -1, -1):
            if isinstance(messages[idx], AIMessage):
                last_ai_idx = idx
                ai = messages[idx]
                break

    tool_done = {
        m.tool_call_id
        for m in messages[last_ai_idx + 1 :]
        if isinstance(m, ToolMessage)
    }
    return [c for c in ai_message_tool_calls(ai) if c.get("id") not in tool_done]


def ai_message_is_thinking_only(msg: AIMessage) -> bool:
    """
    是否 thinking-only：无 visible text、无 pending tool。

    @param msg assistant 消息
    @return 是否仅 thinking
    """
    if ai_message_has_visible_text(msg):
        return False
    return not ai_message_has_tool_calls(msg)


def think_continue_nudge_pending(messages: list[BaseMessage]) -> bool:
    """
    末条是否为 Think step 续跑 nudge（避免重复注入）。

    @param messages 消息列表
    @return 是否已有 nudge
    """
    if len(messages) < 2:
        return False
    last = messages[-1]
    prev = messages[-2]
    if not isinstance(last, HumanMessage) or not isinstance(prev, AIMessage):
        return False
    if str(getattr(last, "content", "") or "").strip() != THINK_CONTINUE_NUDGE.strip():
        return False
    return ai_message_is_thinking_only(prev)


def route_after_agent(
    state: dict,
    *,
    complete_on_thinking_if: Callable[[AIMessage], bool] | None = None,
) -> RouteAfterAgent:
    """
    agent 节点后的路由：有 tool → tools；有 text → END；否则续跑 agent。

    thinking-only（无 text、无 tool）不在此 END，由图内再次调用 agent；
    Plan 子图传入 complete_on_thinking_if 时：仅 visible text 含结构化 JSON 才 END
    （thinking 不算交付，与 Chat 语义一致）。

    @param state ReAct 状态
    @param complete_on_thinking_if Plan 子图：visible 结构化交付物判定（不读 thinking）
    @return 下一节点名或 END
    """
    messages = list(state.get("messages") or [])
    last_ai = last_ai_message(messages)
    if last_ai is None:
        return "__end__"

    if pending_tool_calls(messages, last_ai=last_ai):
        return "tools"

    if complete_on_thinking_if is not None:
        if complete_on_thinking_if(last_ai):
            return "__end__"
    elif ai_message_has_visible_text(last_ai):
        return "__end__"

    remaining = state.get("remaining_steps")
    # 预留 1 步给 turn_fallback 节点本身，避免 recursion_limit 前无法 END
    if remaining is not None and remaining <= 2:
        return "turn_fallback"

    return "think_nudge"


def make_route_after_agent_for_graph(
    *,
    complete_on_thinking_if: Callable[[AIMessage], bool] | None = None,
) -> Callable[[dict], str]:
    """
    构造 LangGraph conditional edge 路由（可选 Plan 结构化 visible END）。

    @param complete_on_thinking_if visible 结构化交付物判定
    @return state → 下一节点名或 END
    """

    def router(state: dict) -> str:
        result = route_after_agent(state, complete_on_thinking_if=complete_on_thinking_if)
        if result == "__end__":
            return END
        return result

    return router


def route_after_agent_for_graph(state: dict) -> str:
    """
    LangGraph conditional edge 用：将 __end__ 映射为 END 常量（Chat Agent 默认）。

    @param state ReAct 状态
    @return 下一节点
    """
    return make_route_after_agent_for_graph()(state)
