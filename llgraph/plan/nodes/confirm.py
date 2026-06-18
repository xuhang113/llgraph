"""Confirm 节点：计划确认 interrupt（approve / revise / cancel）。"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from llgraph.plan.plan_registry import init_plan_session_meta
from llgraph.plan.plan_store import save_plan
from llgraph.plan.runtime import PlanRuntimeContext
from llgraph.plan.state import PlanPhase
from llgraph.plan.workflow_view import build_workflow_snapshot


def _apply_confirm_decision(
    state: dict[str, Any],
    ctx: PlanRuntimeContext,
    decision: dict[str, Any],
) -> dict[str, Any]:
    """
    根据用户决策更新 PlanState。

    @param state PlanState
    @param ctx 运行时上下文
    @param decision approve / revise / cancel 载荷
    @return state 更新
    """
    action = str(decision.get("action") or "approve").strip().lower()
    plan = dict(state.get("plan") or {})
    plan_id = str(plan.get("plan_id") or state.get("plan_id") or "")

    if action in ("cancel", "rejected", "reject"):
        snapshot = build_workflow_snapshot(
            thread_id=ctx.thread_id,
            phase=PlanPhase.CANCELLED,
            plan=plan,
            current_node="confirm",
        )
        return {
            "phase": PlanPhase.CANCELLED,
            "workflow_snapshot": snapshot,
            "cancel_requested": True,
        }

    if action in ("revise", "revision"):
        note = str(
            decision.get("revise_note")
            or decision.get("revision_note")
            or decision.get("note")
            or ""
        ).strip()
        snapshot = build_workflow_snapshot(
            thread_id=ctx.thread_id,
            phase=PlanPhase.PLANNING,
            plan=plan,
            current_node="planner",
        )
        return {
            "phase": PlanPhase.PLANNING,
            "revision_note": note or None,
            "workflow_snapshot": snapshot,
        }

    allow_write = bool(decision.get("allow_worker_write"))
    execution = plan.get("execution") if isinstance(plan.get("execution"), dict) else {}
    execution["allow_worker_write"] = allow_write
    plan["execution"] = execution
    plan["phase"] = PlanPhase.EXECUTING
    if plan_id:
        save_plan(ctx.workspace, plan, plans_dir=ctx.settings.plans_dir)
        init_plan_session_meta(
            ctx.workspace,
            ctx.thread_id,
            plan_id,
            phase=PlanPhase.EXECUTING,
            title=str(plan.get("title") or ""),
        )

    snapshot = build_workflow_snapshot(
        thread_id=ctx.thread_id,
        phase=PlanPhase.EXECUTING,
        plan=plan,
        current_node="supervisor",
    )
    return {
        "phase": PlanPhase.EXECUTING,
        "allow_worker_write": allow_write,
        "plan": plan,
        "revision_note": None,
        "workflow_snapshot": snapshot,
    }


def confirm_node(state: dict[str, Any], ctx: PlanRuntimeContext) -> dict[str, Any]:
    """
    Confirm 节点：interrupt 等待用户确认计划。

    @param state PlanState
    @param ctx 运行时上下文
    @return state 更新
    """
    phase = str(state.get("phase") or "")
    plan = state.get("plan") if isinstance(state.get("plan"), dict) else {}

    if phase != PlanPhase.AWAITING_CONFIRM:
        return {}

    tasks = plan.get("tasks") if isinstance(plan.get("tasks"), list) else []
    payload = {
        "type": "plan_confirm",
        "plan_id": plan.get("plan_id"),
        "title": plan.get("title"),
        "goal": plan.get("goal") or state.get("opening_goal"),
        "task_count": len(tasks),
        "tasks": [
            {"id": str(t.get("id") or ""), "title": str(t.get("title") or "")}
            for t in tasks
            if isinstance(t, dict)
        ],
    }
    decision = interrupt(payload)
    if not isinstance(decision, dict):
        decision = {"action": "approve"}
    return _apply_confirm_decision(state, ctx, decision)
