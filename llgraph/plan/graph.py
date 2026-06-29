"""PlanGraph 编译与路由。

父图 StateGraph(PlanState) 的 planner/worker node 内 invoke 独立 ReAct 子图
（LangGraph 官方「不同 state → node 内 call subgraph」模式），便于后续
对父图 stream(..., subgraphs=True) 扩展 Web 工作流。
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from llgraph.plan.nodes.confirm import confirm_node
from llgraph.plan.nodes.planner import run_planner_node
from llgraph.plan.nodes.supervisor import route_after_supervisor, supervisor_node
from llgraph.plan.nodes.synthesize import synthesize_node
from llgraph.plan.nodes.worker import worker_node
from llgraph.plan.runtime import PlanRuntimeContext
from llgraph.plan.state import PlanPhase, PlanState
from llgraph.plan.subgraph_registry import PLAN_SUBGRAPH_REGISTRY


def _route_after_confirm(state: PlanState) -> str:
    phase = str(state.get("phase") or "")
    if phase == PlanPhase.CANCELLED:
        return "end"
    if phase == PlanPhase.PLANNING:
        return "planner"
    if phase == PlanPhase.EXECUTING:
        return "supervisor"
    return "end"


def build_plan_graph(ctx: PlanRuntimeContext):
    """
    构建并 compile PlanGraph。

    @param ctx 运行时上下文
    @return compiled graph
    """
    from llgraph.core.checkpointer_factory import create_checkpointer

    graph = StateGraph(PlanState)

    def planner_wrapper(state: PlanState) -> dict[str, Any]:
        return run_planner_node(state, ctx)

    def confirm_wrapper(state: PlanState) -> dict[str, Any]:
        return confirm_node(state, ctx)

    def supervisor_wrapper(state: PlanState) -> dict[str, Any]:
        return supervisor_node(state, ctx)

    def worker_wrapper(state: PlanState) -> dict[str, Any]:
        return worker_node(state, ctx)

    def synthesize_wrapper(state: PlanState) -> dict[str, Any]:
        return synthesize_node(state, ctx)

    graph.add_node("planner", planner_wrapper)
    graph.add_node("confirm", confirm_wrapper)
    graph.add_node("supervisor", supervisor_wrapper)
    graph.add_node("worker", worker_wrapper)
    graph.add_node("synthesize", synthesize_wrapper)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "confirm")
    graph.add_conditional_edges(
        "confirm",
        _route_after_confirm,
        {"planner": "planner", "supervisor": "supervisor", "end": END},
    )
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"worker": "worker", "synthesize": "synthesize", "confirm": "confirm", "end": END},
    )
    graph.add_edge("worker", "supervisor")
    graph.add_edge("synthesize", END)

    checkpointer = create_checkpointer(ctx.workspace, with_memory=True, thread_key=ctx.thread_id)
    compiled = graph.compile(checkpointer=checkpointer)
    compiled.plan_subgraph_registry = PLAN_SUBGRAPH_REGISTRY  # type: ignore[attr-defined]
    return compiled
