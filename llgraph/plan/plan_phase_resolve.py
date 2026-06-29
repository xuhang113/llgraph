"""Plan phase 合并解析（plan_state / meta / plan.json）。"""

from __future__ import annotations

from typing import Any

from llgraph.plan.state import PlanPhase


def _phase_rank(phase: str) -> int:
    order = {
        PlanPhase.PLANNING: 0,
        PlanPhase.AWAITING_CONFIRM: 1,
        PlanPhase.EXECUTING: 2,
        PlanPhase.COMPLETED: 3,
        PlanPhase.CANCELLED: 3,
    }
    return order.get(phase, -1)


def task_statuses(plan: dict[str, Any]) -> list[str]:
    """@param plan plan dict @return 各 task status 列表"""
    tasks = plan.get("tasks") if isinstance(plan.get("tasks"), list) else []
    return [str(t.get("status") or "") for t in tasks if isinstance(t, dict)]


def resolve_plan_phase(
    *,
    plan_state: dict[str, Any],
    meta: dict[str, Any],
    plan: dict[str, Any],
) -> str:
    """
    合并 plan_state / meta / plan.json 的 phase，避免 checkpoint 滞后导致 Web 仍显示待确认。

    @param plan_state plan_state.json
    @param meta session meta.json
    @param plan 合并后的 plan dict
    @return 解析后的 phase
    """
    candidates = [
        str(plan_state.get("phase") or "").strip(),
        str(meta.get("phase") or "").strip(),
        str(plan.get("phase") or "").strip(),
    ]
    candidates = [c for c in candidates if c]

    statuses = task_statuses(plan)
    has_progress = any(s in ("running", "done", "failed") for s in statuses)
    all_terminal = bool(statuses) and all(
        s in ("done", "failed", "skipped") for s in statuses
    )

    if all_terminal:
        if str(plan_state.get("final_report") or "").strip():
            return PlanPhase.COMPLETED
        if PlanPhase.COMPLETED in candidates:
            return PlanPhase.COMPLETED
        if PlanPhase.CANCELLED in candidates:
            return PlanPhase.CANCELLED
        if any(s == "skipped" for s in statuses) and not any(s == "done" for s in statuses):
            return PlanPhase.CANCELLED
        if has_progress:
            return PlanPhase.EXECUTING
        return PlanPhase.COMPLETED

    if has_progress:
        best = max(candidates, key=_phase_rank) if candidates else PlanPhase.EXECUTING
        if _phase_rank(best) < _phase_rank(PlanPhase.EXECUTING):
            return PlanPhase.EXECUTING
        return best

    if candidates:
        return max(candidates, key=_phase_rank)
    return "unknown"
