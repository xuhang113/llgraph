"""ReAct 子图基元：兼容层（实现位于 llgraph.subagent）。"""

from __future__ import annotations

from typing import Any

from llgraph.plan.runtime import PlanRuntimeContext
from llgraph.subagent.engine import (
    ReactSubgraphSpec,
    build_react_subgraph,
    collect_subgraph_messages,
    invoke_react_subgraph_sync,
    invoke_react_subgraph_turn as invoke_react_subgraph_turn_shared,
    subgraph_invoke_config,
)
from llgraph.subagent.runtime import SubagentRuntime

__all__ = [
    "ReactSubgraphSpec",
    "build_react_subgraph",
    "collect_subgraph_messages",
    "invoke_react_subgraph_sync",
    "invoke_react_subgraph_turn",
    "subgraph_invoke_config",
]


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
    Plan 适配：PlanRuntimeContext → 共享 SubagentRuntime 后 invoke。

    @param plan_state 父 PlanState（manifest 写权限）
    """
    effective_allow = ctx.subagent_manifest_allow_write(plan_state, allow_write=allow_write)
    task_id = None
    if ":worker:" in sub_thread:
        task_id = sub_thread.rsplit(":worker:", 1)[-1].strip() or None

    def _cancel_check() -> bool:
        from llgraph.plan.execution_coordinator import is_cancel_requested, is_task_cancel_requested

        if is_cancel_requested(ctx.thread_id):
            return True
        if task_id and is_task_cancel_requested(ctx.thread_id, task_id):
            return True
        return False

    max_turns = (
        ctx.settings.planner_max_turns
        if ":planner:" in sub_thread
        else ctx.settings.worker_max_turns
    )
    runtime = SubagentRuntime(
        workspace=ctx.workspace,
        parent_thread_id=ctx.thread_id,
        trace_session=ctx.trace_session,
        context_session=ctx.context_session,
        allow_write=effective_allow,
        mcp_tools=list(ctx.mcp_tools or []),
        sandbox_policy=ctx.sandbox_policy,
        web_search_enabled=ctx.web_search_enabled,
        write_failure_tracker=ctx.write_failure_tracker,
        on_file_changed=ctx.on_file_changed,
        sse_emit=ctx.sse_emit,
        sse_loop=ctx.sse_loop,
        max_turns=max_turns,
        cancel_check=_cancel_check,
    )
    return invoke_react_subgraph_turn_shared(
        runtime,
        subgraph,
        user_message,
        sub_thread=sub_thread,
        role_label=role_label,
        spec=spec,
        allow_write=effective_allow,
        recursion_limit=max_turns,
    )
