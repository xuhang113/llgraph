"""ReAct 子图基元（explore / general 子 Agent）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from llgraph.core.agent_turn import FALLBACK_INCOMPLETE_TURN
from llgraph.core.checkpointer_factory import create_checkpointer
from llgraph.core.react_graph import build_react_graph
from llgraph.context.message_normalize import make_prompt_normalizer
from llgraph.subagent.runtime import (
    SubagentRuntime,
    isolated_subagent_trace,
)


@dataclass(frozen=True)
class ReactSubgraphSpec:
    """
    子图元数据。

    @param node_id 子图标识；Agent 工具侧可用 kind
    @param subgraph_kind explore | general
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
    构建 LangGraph ReAct 子图。

    @param subgraph_kind 子图类型（explore | general）
    """
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
        complete_on_thinking_if=None,
    )


def subgraph_invoke_config(sub_thread: str) -> dict[str, Any]:
    """子图 invoke/stream 用的 RunnableConfig。"""
    return {"configurable": {"thread_id": sub_thread}}


def _extract_last_visible_ai_text(messages: list[Any]) -> str:
    from llgraph.core.llm_response import llm_response_text

    for msg in reversed(messages or []):
        role = getattr(msg, "type", None) or getattr(msg, "role", "")
        if str(role) not in ("ai", "assistant"):
            continue
        text = llm_response_text(msg, fallback_thinking=False).strip()
        if text:
            return text
    return ""


def extract_subagent_result_text(
    messages: list[Any],
    *,
    subgraph_kind: str | None,
) -> str:
    """子图 turn 结束后的交付正文（始终取最后一条可见助手文本）。"""
    return _extract_last_visible_ai_text(messages)


def task_id_from_sub_thread(sub_thread: str) -> str | None:
    for marker in (":worker:", ":explore:", ":subagent:", ":planner:"):
        if marker in sub_thread:
            tail = sub_thread.rsplit(marker, 1)[-1].strip()
            return tail or None
    return None


def invoke_react_subgraph_turn(
    runtime: SubagentRuntime,
    subgraph: Any,
    user_message: str,
    *,
    sub_thread: str,
    role_label: str,
    spec: ReactSubgraphSpec | None = None,
    allow_write: bool | None = None,
    recursion_limit: int | None = None,
) -> str:
    """
    Invoke 子图一轮（终端 /trace + Web SSE）。

    @param runtime 已 fork 的子运行时
    @param subgraph 已 compile 的 ReAct 子图
    @param user_message 本轮任务提示
    @param sub_thread 子图 checkpoint thread_id
    @param role_label 过程展示标题
    @param spec 可选元数据
    @param allow_write manifest/workspace-context 写权限
    @param recursion_limit 覆盖 runtime.max_turns
    @return 助手最终文本
    """
    from llgraph.context.context_builder import (
        build_workspace_context_block,
        wrap_user_message_with_context,
    )
    from llgraph.core.react_limits import resolve_agent_max_turns
    from llgraph.display.trace_display import print_command_prelude, stream_agent_turn
    from llgraph.display.trace_sink import StdoutTraceSink
    from llgraph.session.session_manifest import sync_session_manifest_to_agent_state

    if runtime.trace_session.trace_sink is None:
        runtime.trace_session.trace_sink = StdoutTraceSink()

    kind = spec.subgraph_kind if spec is not None else role_label
    task_id = task_id_from_sub_thread(sub_thread)
    trace = isolated_subagent_trace(
        runtime,
        sub_thread=sub_thread,
        subgraph_kind=kind,
        task_id=task_id,
    )
    if not trace.is_silent():
        detail = "LangGraph 子图 · 工具与推理遵循 /trace 设置"
        if spec is not None:
            detail = f"{spec.subgraph_kind} · {detail}"
        print_command_prelude(f"Subagent · {role_label}", detail=detail)

    effective_allow_write = runtime.allow_write if allow_write is None else bool(allow_write)

    sync_session_manifest_to_agent_state(
        subgraph,
        thread_id=sub_thread,
        workspace=runtime.workspace,
        session=runtime.context_session,
        user_message=user_message,
        with_memory=True,
        allow_write=effective_allow_write,
    )
    context_block = build_workspace_context_block(
        runtime.workspace,
        runtime.context_session,
        user_message,
        allow_write=effective_allow_write,
    )
    effective = wrap_user_message_with_context(user_message, context_block)

    def _cancel_check() -> bool:
        if runtime.cancel_check is not None:
            return bool(runtime.cancel_check())
        return False

    limit = recursion_limit
    if limit is None:
        limit = runtime.max_turns
    if limit is None:
        limit = resolve_agent_max_turns(runtime.workspace)

    turn = stream_agent_turn(
        subgraph,
        user_message,
        thread_id=sub_thread,
        with_memory=True,
        trace_session=trace,
        workspace=runtime.workspace,
        context_session=runtime.context_session,
        effective_message=effective,
        write_failure_tracker=runtime.write_failure_tracker,
        cancel_check=_cancel_check,
        recursion_limit=limit,
    )
    _persist_subagent_web_trace(runtime, sub_thread, trace)

    messages = collect_subgraph_messages(subgraph, sub_thread)
    stream_text = turn.text.strip()
    if stream_text and stream_text != FALLBACK_INCOMPLETE_TURN:
        return stream_text
    if stream_text:
        return stream_text
    return extract_subagent_result_text(messages, subgraph_kind=None)


def _persist_subagent_web_trace(
    runtime: SubagentRuntime,
    sub_thread: str,
    trace: Any,
) -> None:
    if runtime.sse_emit is None:
        return
    sink = trace.trace_sink
    log_lines = list(getattr(sink, "log_lines", []) or [])
    step_payloads: list[dict[str, Any]] = []
    if trace.last_turn_steps:
        from llgraph.console.runtime.sse_sink import _step_to_dict

        step_payloads = [_step_to_dict(s) for s in trace.last_turn_steps]
    if not log_lines and not step_payloads:
        return
    from llgraph.session.web_trace_store import append_web_trace_turn

    append_web_trace_turn(
        runtime.workspace,
        sub_thread,
        log_lines=log_lines,
        steps=step_payloads,
    )


def collect_subgraph_messages(subgraph: Any, sub_thread: str) -> list[Any]:
    """从子图 checkpoint 读取 messages。"""
    config = subgraph_invoke_config(sub_thread)
    try:
        snap = subgraph.get_state(config)
        if snap and snap.values:
            raw = (snap.values or {}).get("messages") or []
            return list(raw)
    except Exception:
        pass
    return []


def invoke_react_subgraph_sync(
    subgraph: Any,
    user_message: str,
    *,
    sub_thread: str,
) -> tuple[str, list[Any]]:
    """同步 invoke 子图（无 /trace）。"""
    config = subgraph_invoke_config(sub_thread)
    result = subgraph.invoke({"messages": [HumanMessage(content=user_message)]}, config)
    messages = list((result or {}).get("messages") or [])
    return _extract_last_visible_ai_text(messages), messages
