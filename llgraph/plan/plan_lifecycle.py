"""Plan 生命周期：终止态、可否调度 Worker、是否待汇总。"""

from __future__ import annotations

from typing import Any

from llgraph.plan.nodes.supervisor import all_tasks_terminal
from llgraph.plan.state import PlanPhase


def is_plan_terminal(state: dict[str, Any]) -> bool:
    """
    Plan 是否已终止（仅汇总完成后）。

    @param state PlanState
    @return 是否不可再调度 Worker
    """
    return str(state.get("phase") or "") == PlanPhase.COMPLETED


def is_plan_closed(state: dict[str, Any]) -> bool:
    """
    Plan 是否已关闭（completed 或 cancelled），不可再 start 新 goal。

    @param state PlanState
    @return 是否 closed
    """
    phase = str(state.get("phase") or "")
    return phase in (PlanPhase.COMPLETED, PlanPhase.CANCELLED)


def can_schedule_workers(state: dict[str, Any]) -> bool:
    """
    是否允许调度/重跑 Work task。

    @param state PlanState
    @return 非 completed/cancelled 且未在仅汇总阶段
    """
    phase = str(state.get("phase") or "")
    if phase in (PlanPhase.COMPLETED, PlanPhase.CANCELLED):
        return False
    if phase not in (PlanPhase.EXECUTING, PlanPhase.AWAITING_CONFIRM, PlanPhase.PLANNING):
        return False
    if phase == PlanPhase.EXECUTING:
        plan = state.get("plan") if isinstance(state.get("plan"), dict) else {}
        if all_tasks_terminal(plan) and not str(state.get("final_report") or "").strip():
            return False
    return phase == PlanPhase.EXECUTING


def needs_synthesize(state: dict[str, Any]) -> bool:
    """
    全部 Work 已终态但尚未生成 final_report，应继续走汇总。

    @param state PlanState
    @return 是否待 Synthesize
    """
    if is_plan_terminal(state):
        return False
    if str(state.get("phase") or "") == PlanPhase.CANCELLED:
        return False
    if str(state.get("final_report") or "").strip():
        return False
    plan = state.get("plan") if isinstance(state.get("plan"), dict) else {}
    return all_tasks_terminal(plan)


def can_discuss(state: dict[str, Any]) -> bool:
    """终止后是否可基于报告问答。"""
    return is_plan_terminal(state) and bool(str(state.get("final_report") or "").strip())


def worker_run_block_reason(state: dict[str, Any], *, action: str = "run") -> str:
    """
    不可调度 Worker 时的原因文案。

    @param state PlanState
    @param action run | continue
    @return 空串表示可执行
    """
    if is_plan_terminal(state):
        return "Plan 已汇总完成并终止，不能再执行 Work；可直接输入问题追问报告，或 /plan revise 重新规划。"
    if str(state.get("phase") or "") == PlanPhase.CANCELLED:
        return "Plan 已取消。"
    if needs_synthesize(state):
        if action == "run":
            return "全部 Work 已完成，请 /plan continue 进入汇总阶段，不能单独重跑 Work。"
        return ""
    phase = str(state.get("phase") or "")
    if phase == PlanPhase.AWAITING_CONFIRM:
        return "计划待确认，请先 /plan confirm。"
    if phase == PlanPhase.PLANNING:
        return "计划生成中或尚未确认，请等待 Planner 完成。"
    if phase != PlanPhase.EXECUTING:
        return f"当前阶段为 {phase}，无法执行 Work。"
    return ""
