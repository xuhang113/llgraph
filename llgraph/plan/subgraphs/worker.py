"""Worker 子 Agent（LangGraph ReAct 子图）。"""

from __future__ import annotations

from typing import Any

from llgraph.core.llm import create_gateway_llm
from llgraph.core.tools import get_agent_tools
from llgraph.plan.runtime import PlanRuntimeContext
from llgraph.plan.subgraph_prompt import build_subagent_system_prompt, build_worker_role_block
from llgraph.plan.subgraphs.base import (
    ReactSubgraphSpec,
    build_react_subgraph,
    collect_subgraph_messages,
    invoke_react_subgraph_turn,
)

WORKER_SUBGRAPH_SPEC = ReactSubgraphSpec(
    node_id="worker",
    subgraph_kind="worker",
    thread_suffix=":worker:{task_id}",
)


def worker_system_prompt(ctx: PlanRuntimeContext, task: dict[str, Any], *, allow_write: bool) -> str:
    """
    Worker 系统提示（含 skills/rules）。

    @param ctx Plan 运行时上下文
    @param task task 定义
    @param allow_write 是否允许写文件
    @return 系统提示文本
    """
    return build_subagent_system_prompt(
        ctx,
        build_worker_role_block(ctx, task, allow_write=allow_write),
        allow_write=allow_write,
    )


def build_worker_subgraph(
    ctx: PlanRuntimeContext,
    task: dict[str, Any],
    *,
    allow_write: bool,
    sub_thread: str,
) -> Any:
    """
    构建 Worker ReAct 子图。

    @param ctx Plan 运行时上下文
    @param task task 定义
    @param allow_write 是否允许写文件
    @param sub_thread 子图 thread（编辑账本 session_id）
    @return 已 compile 的子图
    """
    llm = create_gateway_llm(ctx.workspace)
    edit_tracker = ctx.subagent_edit_tracker(sub_thread, allow_write=allow_write)
    tools = get_agent_tools(
        workspace_root=ctx.workspace,
        allow_write=allow_write,
        mcp_tools=ctx.mcp_tools,
        web_search_enabled=ctx.web_search_enabled,
        sandbox_policy=ctx.sandbox_policy,
        edit_tracker=edit_tracker,
        on_file_changed=ctx.on_file_changed,
        write_failure_tracker=ctx.write_failure_tracker,
    )
    return build_react_subgraph(
        llm,
        tools,
        worker_system_prompt(ctx, task, allow_write=allow_write),
        workspace=ctx.workspace,
        thread_key=sub_thread,
        subgraph_kind="worker",
    )


def run_worker_subagent(
    ctx: PlanRuntimeContext,
    task: dict[str, Any],
    *,
    task_id: str,
    allow_write: bool,
    user_prompt: str,
    plan_state: dict[str, Any] | None = None,
) -> tuple[str, list[Any], list[str]]:
    """
    在父图 worker node 内 invoke Worker 子图。

    @param ctx Plan 运行时上下文
    @param task task 定义
    @param task_id Task ID
    @param allow_write 是否允许写文件
    @param user_prompt 本轮任务提示
    @return (助手文本, checkpoint messages, 变更文件路径)
    """
    sub_thread = f"{ctx.thread_id}{WORKER_SUBGRAPH_SPEC.thread_suffix.format(task_id=task_id)}"
    edit_tracker = ctx.subagent_edit_tracker(sub_thread, allow_write=allow_write)
    subgraph = build_worker_subgraph(ctx, task, allow_write=allow_write, sub_thread=sub_thread)
    text = invoke_react_subgraph_turn(
        ctx,
        subgraph,
        user_prompt,
        sub_thread=sub_thread,
        role_label=f"Worker {task_id}",
        spec=WORKER_SUBGRAPH_SPEC,
        allow_write=allow_write,
        plan_state=plan_state,
    )
    messages = collect_subgraph_messages(subgraph, sub_thread)
    from llgraph.plan.subgraph_messages import collect_and_persist_subgraph_messages

    collect_and_persist_subgraph_messages(
        ctx.workspace,
        ctx.thread_id,
        task_id,
        subgraph,
        sub_thread,
        fallback_messages=messages,
    )
    files_changed = edit_tracker.unique_paths() if edit_tracker is not None else []
    return text, messages, files_changed
