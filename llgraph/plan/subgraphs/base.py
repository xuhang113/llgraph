"""ReAct 子图基元：构建 CompiledStateGraph 并在父图 node 内 invoke。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from llgraph.core.checkpointer_factory import create_checkpointer
from llgraph.context.message_normalize import make_prompt_normalizer
from llgraph.plan.runtime import PlanRuntimeContext


@dataclass(frozen=True)
class ReactSubgraphSpec:
    """
    子图元数据（供 workflow 视图与 Web 流式扩展）。

    @param node_id 父图中挂载该子图的 node id
    @param subgraph_kind 子图类型（planner | worker）
    @param thread_suffix checkpoint thread 后缀模板
    """

    node_id: str
    subgraph_kind: str
    thread_suffix: str


def build_react_subgraph(
    llm: Any,
    tools: list[Any],
    system_prompt: str,
    *,
    workspace: Path,
    with_memory: bool = True,
) -> Any:
    """
    构建 LangGraph ReAct 子图（create_react_agent + checkpointer）。

    @param llm 语言模型
    @param tools 工具列表
    @param system_prompt 系统提示（含 rules/skills 规范）
    @param workspace 工作区根
    @param with_memory 是否启用 checkpoint
    @return 已 compile 的子图
    """
    from llgraph.context.tool_call_repair import repair_tool_calls_post_model_state

    checkpointer = create_checkpointer(workspace, with_memory=with_memory)
    return create_react_agent(
        llm,
        tools,
        prompt=make_prompt_normalizer(system_prompt, workspace),
        checkpointer=checkpointer,
        post_model_hook=repair_tool_calls_post_model_state,
    )


def subgraph_invoke_config(sub_thread: str) -> dict[str, Any]:
    """
    子图 invoke/stream 用的 RunnableConfig。

    @param sub_thread 子图独立 thread_id（如 plan-xxx:planner:v1）
    @return configurable 配置
    """
    return {"configurable": {"thread_id": sub_thread}}


from llgraph.context.message_normalize import _message_text


def _extract_last_ai_text(messages: list[Any]) -> str:
    for msg in reversed(messages or []):
        role = getattr(msg, "type", None) or getattr(msg, "role", "")
        if str(role) in ("ai", "assistant"):
            content = getattr(msg, "content", "")
            return _message_text(content).strip()
    return ""


def invoke_react_subgraph_turn(
    ctx: PlanRuntimeContext,
    subgraph: Any,
    user_message: str,
    *,
    sub_thread: str,
    role_label: str,
    spec: ReactSubgraphSpec | None = None,
) -> str:
    """
    在父图 node 内 invoke 子图一轮（终端沿用 /trace 展示）。

    父图 PlanState 与子图 MessagesState 不同，采用 LangGraph 官方
    「node 内 call subgraph + state 映射」模式；后续 Web 可对父图
    stream(..., subgraphs=True) 订阅嵌套事件。

    @param ctx Plan 运行时上下文
    @param subgraph 已 compile 的 ReAct 子图
    @param user_message 本轮用户/任务提示
    @param sub_thread 子图 checkpoint thread_id
    @param role_label 过程展示标题
    @param spec 可选子图元数据
    @return 助手最终文本
    """
    from llgraph.display.trace_display import print_command_prelude, stream_agent_turn
    from llgraph.display.trace_sink import StdoutTraceSink

    from llgraph.context.context_builder import build_workspace_context_block, wrap_user_message_with_context
    from llgraph.session.session_manifest import sync_session_manifest_to_agent_state

    if ctx.trace_session.trace_sink is None:
        ctx.trace_session.trace_sink = StdoutTraceSink()

    trace = ctx.trace_session
    if not trace.is_silent():
        detail = "LangGraph 子图 · 工具与推理遵循 /trace 设置"
        if spec is not None:
            detail = f"{spec.subgraph_kind} · {detail}"
        print_command_prelude(f"Plan · {role_label}", detail=detail)

    sync_session_manifest_to_agent_state(
        subgraph,
        thread_id=sub_thread,
        workspace=ctx.workspace,
        session=ctx.context_session,
        user_message=user_message,
        with_memory=True,
        allow_write=ctx.allow_write_cli,
    )
    context_block = build_workspace_context_block(
        ctx.workspace,
        ctx.context_session,
        user_message,
        allow_write=ctx.allow_write_cli,
    )
    effective = wrap_user_message_with_context(user_message, context_block)

    turn = stream_agent_turn(
        subgraph,
        user_message,
        thread_id=sub_thread,
        with_memory=True,
        trace_session=trace,
        workspace=ctx.workspace,
        context_session=ctx.context_session,
        effective_message=effective,
        write_failure_tracker=ctx.write_failure_tracker,
    )
    return turn.text


def invoke_react_subgraph_sync(
    subgraph: Any,
    user_message: str,
    *,
    sub_thread: str,
) -> tuple[str, list[Any]]:
    """
    同步 invoke 子图（无 /trace，供批处理或测试）。

    @param subgraph 已 compile 的子图
    @param user_message 用户消息
    @param sub_thread 子图 thread_id
    @return (助手文本, messages)
    """
    config = subgraph_invoke_config(sub_thread)
    result = subgraph.invoke({"messages": [HumanMessage(content=user_message)]}, config)
    messages = list((result or {}).get("messages") or [])
    return _extract_last_ai_text(messages), messages


def collect_subgraph_messages(subgraph: Any, sub_thread: str) -> list[Any]:
    """
    从子图 checkpoint 读取 messages（落盘 subgraphs/<task>/messages.jsonl）。

    @param subgraph 已 compile 的子图
    @param sub_thread 子图 thread_id
    @return messages 列表
    """
    config = subgraph_invoke_config(sub_thread)
    try:
        snap = subgraph.get_state(config)
        if snap and snap.values:
            raw = (snap.values or {}).get("messages") or []
            return list(raw)
    except Exception:
        pass
    return []
