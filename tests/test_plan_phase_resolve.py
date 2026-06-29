"""Plan phase 合并与待汇总判定。"""

from llgraph.plan.plan_lifecycle import needs_synthesize
from llgraph.plan.plan_phase_resolve import resolve_plan_phase
from llgraph.plan.state import PlanPhase


def test_resolve_phase_all_done_no_report_is_executing():
    plan = {"phase": "awaiting_confirm", "tasks": [{"id": "w1", "status": "done"}]}
    phase = resolve_plan_phase(plan_state={"phase": "awaiting_confirm"}, meta={}, plan=plan)
    assert phase == PlanPhase.EXECUTING


def test_resolve_phase_all_done_with_report_is_completed():
    plan = {"phase": "executing", "tasks": [{"id": "w1", "status": "done"}]}
    phase = resolve_plan_phase(
        plan_state={"phase": "executing", "final_report": "# ok"},
        meta={},
        plan=plan,
    )
    assert phase == PlanPhase.COMPLETED


def test_needs_synthesize_when_tasks_done_without_report():
    state = {
        "phase": PlanPhase.AWAITING_CONFIRM,
        "final_report": None,
        "plan": {"tasks": [{"id": "w1", "status": "done"}]},
    }
    assert needs_synthesize(state) is True
