"""ReAct 工具节点：并行执行 + search_code_parallel 去重。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.tool_node import ToolNode

from llgraph.context.chat_history_repair import ai_message_tool_calls
from llgraph.core.code_index_tools import _DUPLICATE_PARALLEL_MSG
from llgraph.core.tool_execution_context import set_tool_execution_messages

_PARALLEL_SEARCH_TOOL = "search_code_parallel"


def _emit_tool_start_milestones(prior: list[BaseMessage]) -> None:
    """工具节点入口：推送 SSE/落盘里程碑，避免长工具阻塞时 Trace 看似卡死。"""
    from llgraph.display.trace_display import LAST_TRACE_SESSION, emit_trace_milestone

    session = LAST_TRACE_SESSION
    if session is None or session.is_silent():
        return
    for msg in reversed(prior):
        if not isinstance(msg, AIMessage):
            continue
        names: list[str] = []
        for call in ai_message_tool_calls(msg):
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name:
                names.append(str(name))
        if names:
            label = " · ".join(names[:3])
            if len(names) > 3:
                label += f" 等{len(names)}个"
            emit_trace_milestone(session, f"正在运行 {label}…")
        break


def _tool_call_name(call: dict[str, Any]) -> str:
    return str(call.get("name") or "").strip()


def _tool_call_id(call: dict[str, Any]) -> str:
    return str(call.get("id") or "").strip()


def guard_parallel_search_code_parallel(
    state: dict[str, Any],
) -> tuple[dict[str, Any], list[ToolMessage]]:
    """
    同一条 AIMessage 内并行多次 search_code_parallel 时，仅保留首次真正执行。

    @param state 图状态
    @return (可能改写后的 state, 被拦截 tool 的占位 ToolMessage)
    """
    msgs = list(state.get("messages") or [])
    blocked: list[ToolMessage] = []
    for idx in range(len(msgs) - 1, -1, -1):
        msg = msgs[idx]
        if not isinstance(msg, AIMessage):
            continue
        calls = ai_message_tool_calls(msg)
        if not calls:
            continue
        seen_parallel = False
        kept_calls: list[dict[str, Any]] = []
        changed = False
        for call in calls:
            item = dict(call) if isinstance(call, dict) else call
            if not isinstance(item, dict):
                kept_calls.append(call)
                continue
            name = _tool_call_name(item)
            if name == _PARALLEL_SEARCH_TOOL:
                cid = _tool_call_id(item)
                if seen_parallel:
                    blocked.append(
                        ToolMessage(
                            content=_DUPLICATE_PARALLEL_MSG,
                            tool_call_id=cid or "parallel-dup",
                            name=name,
                        )
                    )
                    changed = True
                    continue
                seen_parallel = True
            kept_calls.append(item)
        if changed:
            patched = msgs[:idx] + [msg.model_copy(update={"tool_calls": kept_calls})] + msgs[idx + 1 :]
            state = {**state, "messages": patched}
        break
    return state, blocked


def _merge_tool_outputs(out: dict[str, Any], blocked: list[ToolMessage]) -> dict[str, Any]:
    if not blocked:
        return out
    new_msgs = list(blocked) + list(out.get("messages") or [])
    return {**out, "messages": new_msgs}


_AGENT_CANCEL_TOOL_MSG = "[llgraph] 用户已停止当前生成。"


def _agent_cancel_requested() -> bool:
    from llgraph.context.runtime_context import get_active_thread_id

    tid = get_active_thread_id()
    if not tid:
        return False
    from llgraph.console.runtime.agent_service import is_agent_cancel_requested

    return is_agent_cancel_requested(tid)


def _cancel_pending_tool_messages(messages: list[BaseMessage]) -> list[ToolMessage]:
    from llgraph.core.agent_turn import last_ai_message, pending_tool_calls

    ai = last_ai_message(messages)
    if ai is None:
        return []
    blocked: list[ToolMessage] = []
    for call in pending_tool_calls(messages, last_ai=ai):
        cid = str(call.get("id") or "user-stop").strip() or "user-stop"
        name = call.get("name")
        blocked.append(
            ToolMessage(
                content=_AGENT_CANCEL_TOOL_MSG,
                tool_call_id=cid,
                name=str(name) if name else None,
            )
        )
    return blocked


def build_tool_node(
    tools: list[Any],
    *,
    workspace: Path | None = None,
) -> Callable[..., dict[str, Any]]:
    """
    包装 LangGraph ToolNode：并行执行 tool_calls，拦截重复 search_code_parallel。

    @param tools 工具列表
    @param workspace 工作区根（保留参数以兼容调用方）
    @return 图节点可调用对象
    """
    _ = workspace
    inner = ToolNode(tools)

    def invoke(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        prior = list(state.get("messages") or [])
        state, blocked = guard_parallel_search_code_parallel(state)
        if _agent_cancel_requested():
            cancel_msgs = _cancel_pending_tool_messages(list(state.get("messages") or prior))
            if cancel_msgs:
                return {"messages": cancel_msgs}
        set_tool_execution_messages(list(state.get("messages") or prior))
        try:
            _emit_tool_start_milestones(list(state.get("messages") or prior))
            out = inner.invoke(state, config)
            return _merge_tool_outputs(out, blocked)
        finally:
            set_tool_execution_messages(None)

    async def ainvoke(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        prior = list(state.get("messages") or [])
        state, blocked = guard_parallel_search_code_parallel(state)
        if _agent_cancel_requested():
            cancel_msgs = _cancel_pending_tool_messages(list(state.get("messages") or prior))
            if cancel_msgs:
                return {"messages": cancel_msgs}
        set_tool_execution_messages(list(state.get("messages") or prior))
        try:
            _emit_tool_start_milestones(list(state.get("messages") or prior))
            out = await inner.ainvoke(state, config)
            return _merge_tool_outputs(out, blocked)
        finally:
            set_tool_execution_messages(None)

    invoke.__name__ = "tools"
    ainvoke.__name__ = "tools"
    from langgraph._internal._runnable import RunnableCallable

    return RunnableCallable(invoke, ainvoke, name="tools")
