"""Plan 子图 ReAct 执行（兼容层，委托 subgraphs.base）。"""

from __future__ import annotations

from typing import Any

from llgraph.plan.runtime import PlanRuntimeContext
from llgraph.plan.subgraphs.base import (
    collect_subgraph_messages as _collect_subgraph_messages,
    invoke_react_subgraph_turn,
)


def run_plan_subgraph_turn(
    ctx: PlanRuntimeContext,
    agent: Any,
    user_message: str,
    *,
    sub_thread: str,
    role_label: str,
) -> str:
    """
    在 Plan 子图内执行一轮 ReAct，展示规则与 Agent 模式一致（/trace）。

    @param ctx Plan 运行时上下文
    @param agent 已 compile 的 ReAct 子图
    @param user_message 用户/任务提示
    @param sub_thread 子图 checkpointer thread_id
    @param role_label 过程展示标题（Planner / Worker w1 等）
    @return 助手最终文本
    """
    return invoke_react_subgraph_turn(
        ctx,
        agent,
        user_message,
        sub_thread=sub_thread,
        role_label=role_label,
    )


def collect_subgraph_messages(agent: Any, sub_thread: str) -> list[Any]:
    """
    读取子图 checkpoint 中的 messages（用于落盘）。

    @param agent 已 compile 的 ReAct 子图
    @param sub_thread 子图 thread_id
    @return messages 列表
    """
    return _collect_subgraph_messages(agent, sub_thread)
