"""ReAct 子图基元：构建 CompiledStateGraph 并在父图 node 内 invoke。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from llgraph.core.agent_turn import FALLBACK_INCOMPLETE_TURN
from llgraph.core.checkpointer_factory import create_checkpointer
from llgraph.core.react_graph import build_react_graph
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
    thread_key: str | None = None,
    subgraph_kind: str | None = None,
) -> Any:
    """
    构建 LangGraph ReAct 子图（自建 StateGraph + checkpointer）。

    @param llm 语言模型
    @param tools 工具列表
    @param system_prompt 系统提示（含 rules/skills 规范）
    @param workspace 工作区根
    @param with_memory 是否启用 checkpoint
    @param thread_key checkpoint thread 键
    @param subgraph_kind planner | worker；仅 visible 结构化 JSON 时 END
    @return 已 compile 的子图
    """
    from llgraph.plan.subgraphs.routing import resolve_structured_complete_fn

    checkpointer = create_checkpointer(
        workspace,
        with_memory=with_memory,
        thread_key=thread_key,
    )
    return build_react_graph(
        llm,
        tools,
        prompt=make_prompt_normalizer(system_prompt, workspace),
        checkpointer=checkpointer,
        workspace=workspace,
        complete_on_thinking_if=resolve_structured_complete_fn(subgraph_kind),
    )


def subgraph_invoke_config(sub_thread: str) -> dict[str, Any]:
    """
    子图 invoke/stream 用的 RunnableConfig。

    @param sub_thread 子图独立 thread_id（如 plan-xxx:planner:v1）
    @return configurable 配置
    """
    return {"configurable": {"thread_id": sub_thread}}


def _extract_last_visible_ai_text(messages: list[Any]) -> str:
    """
    取最后一条含可见正文（type:text）的 AI 消息。

    不含 thinking 块；用于 ReAct turn 结束后的 plan 解析等结构化输出场景。

    @param messages checkpoint messages
    @return 可见正文
    """
    from llgraph.core.llm_response import llm_response_text

    for msg in reversed(messages or []):
        role = getattr(msg, "type", None) or getattr(msg, "role", "")
        if str(role) not in ("ai", "assistant"):
            continue
        text = llm_response_text(msg, fallback_thinking=False).strip()
        if text:
            return text
    return ""


def _extract_subagent_result_text(
    messages: list[Any],
    *,
    subgraph_kind: str | None,
) -> str:
    """
    子图 turn 结束后的交付正文（与 complete_on_thinking_if 同一语义）。

    @param messages checkpoint messages
    @param subgraph_kind planner | worker | None
    @return 交付正文
    """
    if subgraph_kind in ("planner", "worker"):
        from llgraph.plan.subgraphs.routing import extract_structured_deliverable_text

        return extract_structured_deliverable_text(messages, subgraph_kind=subgraph_kind)
    return _extract_last_visible_ai_text(messages)


def _extract_last_ai_text(messages: list[Any]) -> str:
    return _extract_last_visible_ai_text(messages)


def invoke_react_subgraph_turn(
    ctx: PlanRuntimeContext,
    subgraph: Any,
    user_message: str,
    *,
    sub_thread: str,
    role_label: str,
    spec: ReactSubgraphSpec | None = None,
    allow_write: bool | None = None,
    plan_state: dict[str, Any] | None = None,
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

    trace = ctx.isolated_subagent_trace(
        sub_thread=sub_thread,
        subgraph_kind=spec.subgraph_kind if spec is not None else role_label,
        task_id=_task_id_from_sub_thread(sub_thread),
    )
    if not trace.is_silent():
        detail = "LangGraph 子图 · 工具与推理遵循 /trace 设置"
        if spec is not None:
            detail = f"{spec.subgraph_kind} · {detail}"
        print_command_prelude(f"Plan · {role_label}", detail=detail)

    effective_allow_write = ctx.subagent_manifest_allow_write(plan_state, allow_write=allow_write)

    sync_session_manifest_to_agent_state(
        subgraph,
        thread_id=sub_thread,
        workspace=ctx.workspace,
        session=ctx.context_session,
        user_message=user_message,
        with_memory=True,
        allow_write=effective_allow_write,
    )
    context_block = build_workspace_context_block(
        ctx.workspace,
        ctx.context_session,
        user_message,
        allow_write=effective_allow_write,
    )
    effective = wrap_user_message_with_context(user_message, context_block)

    task_id = _task_id_from_sub_thread(sub_thread)

    def _cancel_check() -> bool:
        from llgraph.plan.execution_coordinator import is_cancel_requested, is_task_cancel_requested

        if is_cancel_requested(ctx.thread_id):
            return True
        if task_id and is_task_cancel_requested(ctx.thread_id, task_id):
            return True
        return False

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
        cancel_check=_cancel_check,
        recursion_limit=(
            ctx.settings.planner_max_turns
            if ":planner:" in sub_thread
            else ctx.settings.worker_max_turns
        ),
    )
    _persist_subagent_web_trace(ctx, sub_thread, trace)
    kind = spec.subgraph_kind if spec is not None else None
    messages = collect_subgraph_messages(subgraph, sub_thread)
    if kind in ("planner", "worker"):
        deliverable = _extract_subagent_result_text(messages, subgraph_kind=kind)
        if deliverable:
            return deliverable
    stream_text = turn.text.strip()
    if stream_text and stream_text != FALLBACK_INCOMPLETE_TURN:
        return stream_text
    if kind in ("planner", "worker"):
        return _extract_subagent_result_text(messages, subgraph_kind=kind)
    if stream_text:
        return stream_text
    return _extract_last_visible_ai_text(messages)


def _task_id_from_sub_thread(sub_thread: str) -> str | None:
    marker = ":worker:"
    if marker not in sub_thread:
        return None
    tail = sub_thread.rsplit(marker, 1)[-1].strip()
    return tail or None


def _persist_subagent_web_trace(
    ctx: PlanRuntimeContext,
    sub_thread: str,
    trace: Any,
) -> None:
    """子 Agent 本轮 trace 落盘到 sub_thread 会话目录（供 Worker Web 回放）。"""
    if ctx.sse_emit is None:
        return
    sink = trace.trace_sink
    log_lines = list(getattr(sink, "log_lines", []) or [])
    step_payloads: list[dict[str, Any]] = []
    if trace.last_turn_steps:
        from llgraph.console.runtime.sse_sink import _step_to_dict

        step_payloads = [_step_to_dict(s) for s in trace.last_turn_steps]
    if not log_lines and not step_payloads:
        return
    from llgraph.session.web_trace_store import save_last_web_trace

    save_last_web_trace(
        ctx.workspace,
        sub_thread,
        log_lines=log_lines,
        steps=step_payloads,
    )


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
