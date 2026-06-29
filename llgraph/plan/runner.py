"""Plan 图执行：加载状态、invoke、interrupt 恢复。"""

from __future__ import annotations

from typing import Any

from langgraph.types import Command

from llgraph.plan.agent_context import load_agent_context_for_plan
from llgraph.plan.plan_registry import init_plan_session_meta
from llgraph.plan.plan_state_store import load_plan_state, save_plan_state
from llgraph.plan.plan_store import empty_plan, is_placeholder_plan_title, load_plan, new_plan_id, save_plan
from llgraph.session.session_meta import get_session_title
from llgraph.plan.runtime import PlanRuntimeContext
from llgraph.plan.state import PlanPhase
from llgraph.plan.task_scheduling import recover_stale_running_tasks


def _estimate_recursion_limit(ctx: PlanRuntimeContext, state: dict[str, Any] | None = None) -> int:
    """按 task 数量估算父图 recursion_limit。"""
    base = 40
    if state and isinstance(state.get("plan"), dict):
        tasks = state["plan"].get("tasks")
        if isinstance(tasks, list):
            base = max(base, 20 + len(tasks) * 12)
    return min(500, base)


def _graph_config(ctx: PlanRuntimeContext, state: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": ctx.thread_id},
        "recursion_limit": _estimate_recursion_limit(ctx, state),
    }


def _sync_meta(ctx: PlanRuntimeContext, state: dict[str, Any]) -> None:
    plan = state.get("plan") if isinstance(state.get("plan"), dict) else {}
    plan_id = str(state.get("plan_id") or plan.get("plan_id") or "")
    if not plan_id:
        return
    plan_title = str(plan.get("title") or "")
    meta_title = get_session_title(ctx.workspace, ctx.thread_id) or ""
    if plan_title and not is_placeholder_plan_title(plan_title, plan_id):
        title_for_meta = plan_title
    elif meta_title:
        title_for_meta = meta_title
    else:
        title_for_meta = plan_title
    init_plan_session_meta(
        ctx.workspace,
        ctx.thread_id,
        plan_id,
        phase=str(state.get("phase") or ""),
        title=title_for_meta,
    )


def merge_plan_checkpoint_state(ctx: PlanRuntimeContext, graph, state: dict[str, Any]) -> dict[str, Any]:
    """
    合并 LangGraph checkpoint 与 plan_state.json（同进程续跑用）。

    @param ctx 运行时上下文
    @param graph compiled PlanGraph
    @param state 磁盘 plan_state
    @return 合并后的 state
    """
    return _merge_checkpoint_state(ctx, graph, state)


def _merge_checkpoint_state(ctx: PlanRuntimeContext, graph, state: dict[str, Any]) -> dict[str, Any]:
    snap = graph.get_state(_graph_config(ctx))
    if not snap or not snap.values:
        return dict(state)

    disk = dict(state)
    checkpoint = dict(snap.values)
    disk_phase = str(disk.get("phase") or "")
    disk_cancelled = disk_phase == PlanPhase.CANCELLED or bool(disk.get("cancel_requested"))

    if disk_cancelled:
        merged = dict(checkpoint)
        merged["phase"] = disk.get("phase", merged.get("phase"))
        merged["cancel_requested"] = bool(disk.get("cancel_requested"))
        if isinstance(disk.get("plan"), dict):
            merged["plan"] = disk["plan"]
        # 磁盘取消态已清空 batch / snapshot；勿让 executing checkpoint 写回
        merged["parallel_batch"] = (
            list(disk["parallel_batch"])
            if isinstance(disk.get("parallel_batch"), list)
            else []
        )
        if isinstance(disk.get("workflow_snapshot"), dict):
            merged["workflow_snapshot"] = disk["workflow_snapshot"]
        return merged

    merged = dict(disk)
    merged.update(checkpoint)
    return merged


def _persist_state(ctx: PlanRuntimeContext, state: dict[str, Any], *, interrupt: Any | None = None) -> dict[str, Any]:
    from llgraph.plan.plan_resume import apply_pending_interrupt_to_state

    merged = apply_pending_interrupt_to_state(state, interrupt)
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
    if goal:
        from llgraph.session.session_meta import ensure_session_title_auto, sync_plan_json_title_from_session_meta

        ensure_session_title_auto(ctx.workspace, ctx.thread_id, goal)
        sync_plan_json_title_from_session_meta(ctx.workspace, ctx.thread_id)
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
    from llgraph.plan.task_results_hydrate import hydrate_task_results

    merged = hydrate_task_results(
        ctx.workspace,
        state,
        plans_dir=ctx.settings.plans_dir,
    )
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
    phase = str(merged.get("phase") or "")
    if phase == PlanPhase.CANCELLED or merged.get("cancel_requested"):
        return _persist_state(ctx, merged)
    from llgraph.plan.execution_coordinator import clear_cancelled_tasks

    clear_cancelled_tasks(ctx.thread_id)
    merged["cancel_requested"] = False
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
    config = _graph_config(ctx, state)
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
    intr = _extract_interrupt(graph, ctx)
    new_state = _persist_state(ctx, new_state, interrupt=intr)
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
    from llgraph.plan.plan_resume import graph_has_interrupt, resume_after_confirm_without_checkpoint

    config = _graph_config(ctx, state)
    if graph_has_interrupt(graph, config):
        return run_until_interrupt(ctx, graph, state, input_payload=Command(resume=decision))
    return resume_after_confirm_without_checkpoint(
        ctx,
        graph,
        state,
        decision,
        run_until_interrupt=run_until_interrupt,
        resume_executing_plan=resume_executing_plan,
        persist_state=_persist_state,
    )


def resume_executing_plan(
    ctx: PlanRuntimeContext,
    graph,
    state: dict[str, Any],
) -> tuple[dict[str, Any], Any | None]:
    """
    从 executing 阶段继续跑 Supervisor → Worker 链；若仅差汇总则直接 Synthesize。

    @param ctx 运行时上下文
    @param graph compiled graph
    @param state PlanState
    @return (新 state, interrupt)
    """
    from llgraph.plan.plan_lifecycle import needs_synthesize
    from llgraph.plan.execution_coordinator import is_cancel_requested
    from llgraph.plan.nodes.synthesize import synthesize_node

    merged = prepare_plan_for_resume(ctx, state)
    if is_cancel_requested(ctx.thread_id) or merged.get("cancel_requested"):
        merged["phase"] = PlanPhase.CANCELLED
        merged = _persist_state(ctx, merged)
        return merged, None
    force_raw = merged.get("force_task_ids")
    has_force = isinstance(force_raw, list) and any(str(x).strip() for x in force_raw)
    if needs_synthesize(merged) and not has_force:
        updates = synthesize_node(merged, ctx)
        merged = {**merged, **updates}
        merged = _persist_state(ctx, merged)
        return merged, None
    if merged.get("phase") not in (PlanPhase.EXECUTING, PlanPhase.AWAITING_CONFIRM):
        merged["phase"] = PlanPhase.EXECUTING
        merged = _persist_state(ctx, merged)
    config = _graph_config(ctx, merged)
    snap = graph.get_state(config)
    if snap and not snap.next:
        graph.update_state(config, merged, as_node="supervisor")
        return run_until_interrupt(ctx, graph, merged, input_payload=None)
    return run_until_interrupt(ctx, graph, merged)
