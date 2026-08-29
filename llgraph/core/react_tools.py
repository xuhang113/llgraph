"""ReAct 工具节点：并行执行 + 重复工具短路径 + 单工具轮次批量提示。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.tool_node import ToolNode

from llgraph.context.chat_history_repair import ai_message_tool_calls
from llgraph.core.code_index_tools import _DUPLICATE_PARALLEL_MSG
from llgraph.core.react_limits import (
    resolve_batch_tools_nudge_after,
    resolve_identical_tool_guard,
)
from llgraph.core.tool_execution_context import set_tool_execution_messages
from llgraph.core.tool_invoke_timing import (
    attach_tool_timings_to_output,
    reset_tool_timings,
    wrap_tool_node_with_timing,
)
from llgraph.core.tool_loop_guard import (
    clear_tool_loop_guard,
    install_tool_loop_guard,
    wrap_tool_node_with_loop_guard,
)
from llgraph.core.tool_arg_coerce import wrap_tool_node_with_arg_coerce
from llgraph.core.write_serialize import (
    clear_write_serialize_gate,
    install_write_serialize_gate,
    wrap_tool_node_with_write_serialize,
)

_PARALLEL_SEARCH_TOOL = "search_code_parallel"

_BATCH_HINT = (
    "\n\n[llgraph] 已连续多轮每次仅 1 个工具调用。"
    "请在下一条 assistant **同一条消息内并行多个工具调用**——"
    "例如多个 grep_files，或 grep_files + read_files；能合并的勿再拆成多轮。"
)
_BATCH_HINT_MARKER = "[llgraph] 已连续多轮每次仅 1 个工具调用。"


def count_single_tool_rounds_since_user(messages: list[BaseMessage]) -> int:
    """
    统计自最近 user 消息以来，仅含 1 个 tool_call 的 assistant 轮数。

    中间若出现一轮多 tool_call，计数归零（自该点往回不计——从尾部向前：
    遇到 multi 立即返回 0；遇到真实 human 停止）。
    """
    from llgraph.context.investigate_harness import is_ephemeral_harness_human

    count = 0
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            if is_ephemeral_harness_human(msg):
                continue
            break
        if isinstance(msg, AIMessage):
            calls = ai_message_tool_calls(msg)
            if len(calls) == 1:
                count += 1
            elif len(calls) > 1:
                return 0
    return count


def maybe_append_batch_tools_hint(
    out: dict[str, Any],
    *,
    prior_messages: list[BaseMessage],
    workspace: Path | None,
) -> dict[str, Any]:
    """
    连续单工具轮达到阈值时，在本批最后一条 ToolMessage 末尾追加并行提示。

    @param out ToolNode 输出
    @param prior_messages 工具执行前的 state 消息（含本轮 AI tool_calls）
    @param workspace 工作区
    @return 可能改写后的 out
    """
    threshold = resolve_batch_tools_nudge_after(workspace)
    if threshold <= 0:
        return out
    streak = count_single_tool_rounds_since_user(prior_messages)
    if streak < threshold:
        return out
    new_msgs = list(out.get("messages") or [])
    for idx in range(len(new_msgs) - 1, -1, -1):
        msg = new_msgs[idx]
        if not isinstance(msg, ToolMessage):
            continue
        body = str(msg.content or "")
        if _BATCH_HINT_MARKER in body:
            return out
        new_msgs[idx] = ToolMessage(
            content=body + _BATCH_HINT,
            tool_call_id=msg.tool_call_id,
            name=msg.name,
        )
        return {**out, "messages": new_msgs}
    return out


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
    from llgraph.core.react_invoke import agent_cancel_requested

    return agent_cancel_requested()


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
    包装 LangGraph ToolNode：并行执行 tool_calls，拦截重复 search_code_parallel；
    本问内相同参数的 read/grep/失败写短路径返回；同 path 写工具与 read_file 按顺序串行；
    执行前纠偏 Claude Code / Cursor 字段名与宽松类型。

    @param tools 工具列表
    @param workspace 工作区根（保留参数以兼容调用方）
    @return 图节点可调用对象
    """
    inner = ToolNode(tools)
    wrap_tool_node_with_loop_guard(inner)
    wrap_tool_node_with_timing(inner)
    wrap_tool_node_with_write_serialize(inner)
    wrap_tool_node_with_arg_coerce(inner)

    def invoke(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        from llgraph.context.investigate_harness import (
            guard_soft_close_tools,
        )

        prior = list(state.get("messages") or [])
        state, soft_blocked = guard_soft_close_tools(state, workspace=workspace)
        if soft_blocked:
            # SoftClose：清空 tool_calls 后不可再执行
            last_ai = None
            for m in reversed(list(state.get("messages") or [])):
                if isinstance(m, AIMessage):
                    last_ai = m
                    break
            remaining = ai_message_tool_calls(last_ai) if last_ai is not None else []
            if not remaining:
                return {"messages": soft_blocked}
        state, blocked = guard_parallel_search_code_parallel(state)
        blocked = [*soft_blocked, *blocked]
        if _agent_cancel_requested():
            cancel_msgs = _cancel_pending_tool_messages(list(state.get("messages") or prior))
            if cancel_msgs:
                return {"messages": cancel_msgs}
        set_tool_execution_messages(list(state.get("messages") or prior))
        try:
            reset_tool_timings()
            _emit_tool_start_milestones(list(state.get("messages") or prior))
            last_ai = None
            for m in reversed(list(state.get("messages") or [])):
                if isinstance(m, AIMessage):
                    last_ai = m
                    break
            remaining = ai_message_tool_calls(last_ai) if last_ai is not None else []
            if not remaining:
                return {"messages": blocked} if blocked else {"messages": []}
            install_write_serialize_gate(inner, remaining)
            install_tool_loop_guard(
                inner,
                list(state.get("messages") or prior),
                remaining,
                enabled=resolve_identical_tool_guard(workspace),
            )
            try:
                out = attach_tool_timings_to_output(inner.invoke(state, config))
            finally:
                clear_write_serialize_gate(inner)
                clear_tool_loop_guard(inner)
            out = maybe_append_batch_tools_hint(
                out,
                prior_messages=list(state.get("messages") or prior),
                workspace=workspace,
            )
            return _merge_tool_outputs(out, blocked)
        finally:
            set_tool_execution_messages(None)

    async def ainvoke(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        from llgraph.context.investigate_harness import (
            guard_soft_close_tools,
        )

        prior = list(state.get("messages") or [])
        state, soft_blocked = guard_soft_close_tools(state, workspace=workspace)
        if soft_blocked:
            last_ai = None
            for m in reversed(list(state.get("messages") or [])):
                if isinstance(m, AIMessage):
                    last_ai = m
                    break
            remaining = ai_message_tool_calls(last_ai) if last_ai is not None else []
            if not remaining:
                return {"messages": soft_blocked}
        state, blocked = guard_parallel_search_code_parallel(state)
        blocked = [*soft_blocked, *blocked]
        if _agent_cancel_requested():
            cancel_msgs = _cancel_pending_tool_messages(list(state.get("messages") or prior))
            if cancel_msgs:
                return {"messages": cancel_msgs}
        set_tool_execution_messages(list(state.get("messages") or prior))
        try:
            reset_tool_timings()
            _emit_tool_start_milestones(list(state.get("messages") or prior))
            last_ai = None
            for m in reversed(list(state.get("messages") or [])):
                if isinstance(m, AIMessage):
                    last_ai = m
                    break
            remaining = ai_message_tool_calls(last_ai) if last_ai is not None else []
            if not remaining:
                return {"messages": blocked} if blocked else {"messages": []}
            install_write_serialize_gate(inner, remaining)
            install_tool_loop_guard(
                inner,
                list(state.get("messages") or prior),
                remaining,
                enabled=resolve_identical_tool_guard(workspace),
            )
            try:
                out = attach_tool_timings_to_output(await inner.ainvoke(state, config))
            finally:
                clear_write_serialize_gate(inner)
                clear_tool_loop_guard(inner)
            out = maybe_append_batch_tools_hint(
                out,
                prior_messages=list(state.get("messages") or prior),
                workspace=workspace,
            )
            return _merge_tool_outputs(out, blocked)
        finally:
            set_tool_execution_messages(None)

    invoke.__name__ = "tools"
    ainvoke.__name__ = "tools"
    from langgraph._internal._runnable import RunnableCallable

    return RunnableCallable(invoke, ainvoke, name="tools")
