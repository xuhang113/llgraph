"""Plan 图执行：加载状态、invoke、interrupt 恢复。"""

from __future__ import annotations

from typing import Any

from langgraph.types import Command

from llgraph.plan.agent_context import load_agent_context_for_plan
from llgraph.plan.plan_registry import init_plan_session_meta
from llgraph.plan.plan_state_store import load_plan_state, save_plan_state
from llgraph.plan.plan_store import empty_plan, load_plan, new_plan_id, save_plan
from llgraph.plan.runtime import PlanRuntimeContext
from llgraph.plan.state import PlanPhase
from llgraph.plan.task_scheduling import recover_stale_running_tasks


def _graph_config(ctx: PlanRuntimeContext) -> dict[str, Any]:
    return {"configurable": {"thread_id": ctx.thread_id}}


def _sync_meta(ctx: PlanRuntimeContext, state: dict[str, Any]) -> None:
    plan = state.get("plan") if isinstance(state.get("plan"), dict) else {}
    plan_id = str(state.get("plan_id") or plan.get("plan_id") or "")
    if not plan_id:
        return
    init_plan_session_meta(
        ctx.workspace,
        ctx.thread_id,
        plan_id,
        phase=str(state.get("phase") or ""),
        title=str(plan.get("title") or ""),
    )


def _merge_checkpoint_state(ctx: PlanRuntimeContext, graph, state: dict[str, Any]) -> dict[str, Any]:
    snap = graph.get_state(_graph_config(ctx))
    if snap and snap.values:
        merged = dict(state)
        merged.update(dict(snap.values))
        return merged
    return dict(state)


def _persist_state(ctx: PlanRuntimeContext, state: dict[str, Any]) -> dict[str, Any]:
    merged = dict(state)
    save_plan_state(ctx.workspace, ctx.thread_id, merged)
    plan = merged.get("plan") if isinstance(merged.get("plan"), dict) else {}
    plan_id = str(merged.get("plan_id") or plan.get("plan_id") or "")
    if plan_id and plan:
        save_plan(ctx.workspace, plan, plans_dir=ctx.settings.plans_dir)
    _sync_meta(ctx, merged)
    return merged


def _extract_interrupt(graph, ctx: PlanRuntimeContext) -> Any | None:
    snap = graph.get_state(_graph_config(ctx))
    if not snap or not snap.interrupts:
        return None
    first = snap.interrupts[0]
    return getattr(first, "value", first)


def load_or_create_state(
    ctx: PlanRuntimeContext,
    graph,
    *,
    opening_goal: str,
    plan_id: str | None = None,
    step_confirm_each_task: bool = False,
    source_agent_thread_id: str = "",
    agent_context: str = "",
) -> tuple[dict[str, Any], bool]:
    """
    加载或初始化 PlanState。

    @param ctx 运行时上下文
    @param graph compiled PlanGraph
    @param opening_goal 用户目标
    @param plan_id 可选已有 plan_id
    @param step_confirm_each_task 是否每 task interrupt
    @param source_agent_thread_id 来源 Agent thread
    @param agent_context 预置 Agent 上下文
    @return (state, is_new)
    """
    existing = load_plan_state(ctx.workspace, ctx.thread_id)
    if existing:
        state = _merge_checkpoint_state(ctx, graph, existing)
        return state, False

    pid = plan_id or new_plan_id()
    goal = opening_goal.strip()
    agent_ctx = agent_context.strip()
    if not agent_ctx and source_agent_thread_id.strip():
        agent_ctx = load_agent_context_for_plan(ctx.workspace, source_agent_thread_id.strip())

    plan = empty_plan(plan_id=pid, title="", goal=goal)
    plan["phase"] = PlanPhase.PLANNING
    save_plan(ctx.workspace, plan, plans_dir=ctx.settings.plans_dir)

    state: dict[str, Any] = {
        "plan_id": pid,
        "plan": plan,
        "phase": PlanPhase.PLANNING,
        "opening_goal": goal,
        "task_results": {},
        "user_messages": [],
        "final_report": None,
        "error": None,
        "workflow_snapshot": {},
        "allow_worker_write": False,
        "source_agent_thread_id": source_agent_thread_id.strip() or None,
        "agent_context": agent_ctx or None,
        "plan_version": 1,
        "revision_note": None,
        "step_confirm_each_task": step_confirm_each_task,
        "parallel_batch": [],
        "cancel_requested": False,
    }
    state = _persist_state(ctx, state)
    init_plan_session_meta(
        ctx.workspace,
        ctx.thread_id,
        pid,
        phase=PlanPhase.PLANNING,
        title="",
    )
    return state, True


def prepare_plan_for_resume(ctx: PlanRuntimeContext, state: dict[str, Any]) -> dict[str, Any]:
    """
    续跑前恢复僵死 running task，并写回 plan.json。

    @param ctx 运行时上下文
    @param state PlanState
    @return 更新后的 state
    """
    merged = dict(state)
    plan = dict(merged.get("plan") or {})
    plan_id = str(plan.get("plan_id") or merged.get("plan_id") or "")
    if plan_id:
        disk = load_plan(ctx.workspace, plan_id, plans_dir=ctx.settings.plans_dir)
        if disk:
            plan = disk
    plan, changed = recover_stale_running_tasks(plan)
    if changed and plan_id:
        save_plan(ctx.workspace, plan, plans_dir=ctx.settings.plans_dir)
    merged["plan"] = plan
    return _persist_state(ctx, merged)


def run_until_interrupt(
    ctx: PlanRuntimeContext,
    graph,
    state: dict[str, Any],
    *,
    input_payload: Any | None = None,
) -> tuple[dict[str, Any], Any | None]:
    """
    执行 Plan 图直至结束或 interrupt。

    @param ctx 运行时上下文
    @param graph compiled graph
    @param state 当前 PlanState（新建时可传完整 state）
    @param input_payload None、完整 state 或 Command(resume=...)
    @return (新 state, interrupt payload)
    """
    config = _graph_config(ctx)
    if input_payload is None:
        payload: Any = dict(state)
    else:
        payload = input_payload

    try:
        graph.invoke(payload, config)
    except Exception:
        snap = graph.get_state(config)
        if not snap or not snap.interrupts:
            raise

    snap = graph.get_state(config)
    new_state = dict(snap.values) if snap and snap.values else dict(state)
    new_state = _persist_state(ctx, new_state)
    intr = _extract_interrupt(graph, ctx)
    return new_state, intr


def resume_after_confirm_decision(
    ctx: PlanRuntimeContext,
    graph,
    state: dict[str, Any],
    decision: dict[str, Any],
) -> tuple[dict[str, Any], Any | None]:
    """
    从 Confirm interrupt 恢复。

    @param ctx 运行时上下文
    @param graph compiled graph
    @param state PlanState
    @param decision 用户确认决策
    @return (新 state, interrupt)
    """
    return run_until_interrupt(ctx, graph, state, input_payload=Command(resume=decision))


def resume_executing_plan(
    ctx: PlanRuntimeContext,
    graph,
    state: dict[str, Any],
) -> tuple[dict[str, Any], Any | None]:
    """
    从 executing 阶段继续跑 Supervisor → Worker 链。

    @param ctx 运行时上下文
    @param graph compiled graph
    @param state PlanState
    @return (新 state, interrupt)
    """
    merged = prepare_plan_for_resume(ctx, state)
    if merged.get("phase") not in (PlanPhase.EXECUTING, PlanPhase.AWAITING_CONFIRM):
        merged["phase"] = PlanPhase.EXECUTING
        merged = _persist_state(ctx, merged)
    return run_until_interrupt(ctx, graph, merged)
