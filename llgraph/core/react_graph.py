"""自建 ReAct StateGraph：可见正文终止 + tool_call 修复 + Think step 续跑。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph._internal._runnable import RunnableCallable
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt.chat_agent_executor import (
    PROMPT_RUNNABLE_NAME,
    _validate_chat_history,
)
from llgraph.core.react_tools import build_tool_node
from langgraph.types import Checkpointer
from typing_extensions import NotRequired, TypedDict

from llgraph.adapters.inbound import normalize_ai_response
from llgraph.core.agent_turn import (
    FALLBACK_INCOMPLETE_TURN,
    THINK_CONTINUE_NUDGE,
    make_route_after_agent_for_graph,
    think_continue_nudge_pending,
)
from llgraph.context.message_canonical import persist_ai_thinking_in_message


class ReactAgentState(TypedDict):
    """ReAct Agent 图状态（与 LangGraph AgentState 兼容）。"""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_steps: NotRequired[RemainingSteps]


def _prompt_runnable(prompt: Callable[..., Any]) -> RunnableCallable:
    """
    将 make_prompt_normalizer 返回值包装为 LangGraph Prompt runnable。

    @param prompt 接收 state、返回 messages 的可调用对象
    @return Prompt runnable
    """
    return RunnableCallable(prompt, name=PROMPT_RUNNABLE_NAME)


def _bind_tools_if_needed(model: Any, tools: Sequence[BaseTool | Callable | dict]) -> Any:
    """
    为 Chat 模型绑定工具（未绑定时）。

    @param model LLM
    @param tools 工具列表
    @return 可能 bind_tools 后的模型
    """
    if not tools:
        return model
    if isinstance(model, BaseChatModel):
        try:
            return model.bind_tools(
                list(tools),
                parallel_tool_calls=True,
                tool_choice="auto",
            )
        except TypeError:
            try:
                return model.bind_tools(list(tools), parallel_tool_calls=True)
            except TypeError:
                return model.bind_tools(list(tools))
    return model


def _needs_step_limit_message(state: ReactAgentState, response: AIMessage) -> bool:
    """
    步数不足时是否应替换为固定提示（对齐 prebuilt _are_more_steps_needed）。

    @param state 图状态
    @param response 本轮 LLM 响应
    @return 是否替换
    """
    if not response.tool_calls:
        return False
    remaining = state.get("remaining_steps")
    if remaining is None:
        return False
    if remaining < 2:
        return True
    return False


def build_react_graph(
    model: Any,
    tools: Sequence[BaseTool | Callable | dict],
    *,
    prompt: Callable[..., Any],
    checkpointer: Checkpointer | None = None,
    workspace: Path | None = None,
    name: str = "react_agent",
    complete_on_thinking_if: Callable[[AIMessage], bool] | None = None,
) -> CompiledStateGraph:
    """
    构建 ReAct StateGraph（替代 create_react_agent）。

    终止语义：pending tool_calls → tools；有可见 text → END（Chat）；
    thinking-only → 回 agent 续跑（Cursor Think step）；
    Plan 子图传 complete_on_thinking_if：仅 visible text 含结构化 JSON 才 END。

    @param model LangChain Chat 模型
    @param tools 工具列表
    @param prompt make_prompt_normalizer 等 state → messages 可调用对象
    @param checkpointer 可选 checkpoint
    @param workspace 工作区根（入站 normalize 按模型 profile）
    @param name 编译图名称
    @param complete_on_thinking_if Plan 子图：visible 结构化交付物判定（不读 thinking）
    @return 已 compile 的图
    """
    ws = workspace
    if ws is None:
        ws = getattr(model, "llgraph_workspace", None)
    tool_node = build_tool_node(list(tools), workspace=ws)
    bound_model = _bind_tools_if_needed(model, tools)
    if ws is not None:
        from llgraph.core.prompt_cache import apply_prompt_cache_to_llm
        from llgraph.core.prompt_cache_settings import (
            prompt_cache_enabled_for_model,
            resolve_prompt_cache_settings,
        )
        from llgraph.core.llm_settings import resolve_effective_model

        cache_settings = resolve_prompt_cache_settings(ws)
        model_id = resolve_effective_model(ws)
        if (
            prompt_cache_enabled_for_model(ws, model_id)
            and cache_settings.enabled
            and cache_settings.tag_conversation_tail
        ):
            bound_model = apply_prompt_cache_to_llm(bound_model, ws)
    agent_runnable = _prompt_runnable(prompt) | bound_model

    def _normalize_response(response: AIMessage) -> AIMessage:
        from llgraph.core.llm_settings import resolve_effective_model

        model_id = resolve_effective_model(ws)
        return normalize_ai_response(response, ws, model_id)

    def call_agent(state: ReactAgentState, config: RunnableConfig) -> dict[str, Any]:
        messages = list(state.get("messages") or [])
        _validate_chat_history(messages)
        response = agent_runnable.invoke(state, config)
        if not isinstance(response, AIMessage):
            response = AIMessage(content=str(response))

        if _needs_step_limit_message(state, response):
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }

        repaired = persist_ai_thinking_in_message(_normalize_response(response))[0]
        return {"messages": [repaired]}

    async def acall_agent(state: ReactAgentState, config: RunnableConfig) -> dict[str, Any]:
        messages = list(state.get("messages") or [])
        _validate_chat_history(messages)
        response = await agent_runnable.ainvoke(state, config)
        if not isinstance(response, AIMessage):
            response = AIMessage(content=str(response))

        if _needs_step_limit_message(state, response):
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }

        repaired = persist_ai_thinking_in_message(_normalize_response(response))[0]
        return {"messages": [repaired]}

    def think_nudge(state: ReactAgentState) -> dict[str, Any]:
        messages = list(state.get("messages") or [])
        if think_continue_nudge_pending(messages):
            return {}
        return {"messages": [HumanMessage(content=THINK_CONTINUE_NUDGE)]}

    def turn_fallback(_state: ReactAgentState) -> dict[str, Any]:
        return {"messages": [AIMessage(content=FALLBACK_INCOMPLETE_TURN)]}

    workflow = StateGraph(ReactAgentState)
    workflow.add_node("agent", RunnableCallable(call_agent, acall_agent))
    workflow.add_node("tools", tool_node)
    workflow.add_node("think_nudge", think_nudge)
    workflow.add_node("turn_fallback", turn_fallback)

    workflow.set_entry_point("agent")

    route_after_agent = make_route_after_agent_for_graph(
        complete_on_thinking_if=complete_on_thinking_if,
    )
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "think_nudge": "think_nudge",
            "turn_fallback": "turn_fallback",
            END: END,
        },
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("think_nudge", "agent")
    workflow.add_edge("turn_fallback", END)

    return workflow.compile(checkpointer=checkpointer, name=name)
