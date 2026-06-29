"""Planner 子 Agent（LangGraph ReAct 子图）。"""

from __future__ import annotations

from typing import Any

from llgraph.core.llm import create_gateway_llm
from llgraph.core.tools import get_agent_tools
from llgraph.plan.runtime import PlanRuntimeContext
from llgraph.plan.subgraph_prompt import build_planner_role_block, build_subagent_system_prompt
from llgraph.plan.subgraphs.base import (
    ReactSubgraphSpec,
    build_react_subgraph,
    invoke_react_subgraph_turn,
)

PLANNER_SUBGRAPH_SPEC = ReactSubgraphSpec(
    node_id="planner",
    subgraph_kind="planner",
    thread_suffix=":planner:v{version}",
)


def planner_system_prompt(ctx: PlanRuntimeContext) -> str:
    """
    Planner 系统提示（含 skills/rules）。

    @param ctx Plan 运行时上下文
    @return 系统提示文本
    """
    return build_subagent_system_prompt(
        ctx,
        build_planner_role_block(ctx),
        allow_write=False,
    )


def build_planner_subgraph(ctx: PlanRuntimeContext, *, sub_thread: str) -> Any:
    """
    构建 Planner ReAct 子图。

    @param ctx Plan 运行时上下文
    @param sub_thread 子图 checkpoint thread
    @return 已 compile 的子图
    """
    llm = create_gateway_llm(ctx.workspace)
    tools = get_agent_tools(
        workspace_root=ctx.workspace,
        allow_write=False,
        mcp_tools=ctx.mcp_tools,
        web_search_enabled=ctx.web_search_enabled,
        sandbox_policy=ctx.sandbox_policy,
    )
    return build_react_subgraph(
        llm,
        tools,
        planner_system_prompt(ctx),
        workspace=ctx.workspace,
        thread_key=sub_thread,
        subgraph_kind="planner",
    )


def run_planner_subagent(
    ctx: PlanRuntimeContext,
    *,
    user_prompt: str,
    version: int,
    plan_state: dict[str, Any] | None = None,
) -> str:
    """
    在父图 planner node 内 invoke Planner 子图。

    @param ctx Plan 运行时上下文
    @param user_prompt 本轮任务提示
    @param version 计划版本号（用于子 thread）
    @param plan_state 父 PlanState（manifest 用）
    @return 助手最终文本（子图 ReAct turn 结束后的交付正文）
    """
    sub_key = f"planner-v{version}"
    sub_thread = f"{ctx.thread_id}{PLANNER_SUBGRAPH_SPEC.thread_suffix.format(version=version)}"
    subgraph = build_planner_subgraph(ctx, sub_thread=sub_thread)
    planner_ctx = ctx.fork_subagent_runtime(
        sub_thread=sub_thread,
        subgraph_kind="planner",
    )
    text = invoke_react_subgraph_turn(
        planner_ctx,
        subgraph,
        user_prompt,
        sub_thread=sub_thread,
        role_label="Planner",
        spec=PLANNER_SUBGRAPH_SPEC,
        allow_write=False,
        plan_state=plan_state,
    )
    from llgraph.plan.subgraph_messages import collect_and_persist_subgraph_messages
    from llgraph.plan.subgraphs.base import collect_subgraph_messages

    messages = collect_subgraph_messages(subgraph, sub_thread)
    collect_and_persist_subgraph_messages(
        ctx.workspace,
        ctx.thread_id,
        sub_key,
        subgraph,
        sub_thread,
        fallback_messages=messages,
    )
    return text
