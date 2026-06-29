"""Supervisor 节点：确定性 task 调度。"""

from __future__ import annotations

from typing import Any

from llgraph.plan.execution_coordinator import is_cancel_requested
from llgraph.plan.plan_store import load_plan, save_plan
from llgraph.plan.runtime import PlanRuntimeContext
from llgraph.plan.state import PlanPhase, TaskStatus
from llgraph.plan.task_scheduling import (
    deps_satisfied,
    done_task_ids,
    pick_forced_tasks,
    recover_stale_running_tasks,
)
from llgraph.plan.workflow_view import build_workflow_snapshot


def _task_list(plan: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = plan.get("tasks") if isinstance(plan.get("tasks"), list) else []
    return [t for t in tasks if isinstance(t, dict)]


def _find_task(plan: dict[str, Any], task_id: str) -> dict[str, Any] | None:
    key = task_id.strip()
    for task in _task_list(plan):
        if str(task.get("id") or "") == key:
            return task
    return None


def _failed_retries_exhausted(task: dict[str, Any]) -> bool:
    status = str(task.get("status") or TaskStatus.PENDING)
    if status != TaskStatus.FAILED:
        return False
    retry = int(task.get("retry_count") or 0)
    max_retries = int(task.get("max_retries") or 2)
    return retry > max_retries


def has_blocking_failures(plan: dict[str, Any]) -> bool:
    """
    是否存在无法自动重试、且阻塞后续 task 的失败。

    @param plan plan.json
    @return 是否需人工 /plan revise
    """
    for task in _task_list(plan):
        if _failed_retries_exhausted(task):
            return True
        status = str(task.get("status") or TaskStatus.PENDING)
        if status != TaskStatus.PENDING:
            continue
        deps = task.get("depends_on")
        if not isinstance(deps, list):
            continue
        for dep in deps:
            dep_task = _find_task(plan, str(dep))
            if dep_task is not None and _failed_retries_exhausted(dep_task):
                return True
    return False


def all_tasks_terminal(plan: dict[str, Any]) -> bool:
    """
    全部 Work task 是否已终态（done / skipped / 失败且重试用尽）。

    @param plan plan.json
    @return 是否可进入 synthesize
    """
    tasks = _task_list(plan)
    if not tasks:
        return True
    for task in tasks:
        status = str(task.get("status") or TaskStatus.PENDING)
        if status in (TaskStatus.DONE, TaskStatus.SKIPPED):
            continue
        if status == TaskStatus.FAILED and _failed_retries_exhausted(task):
            continue
        return False
    return True


def pick_next_tasks(
    plan: dict[str, Any],
    *,
    max_parallel: int = 1,
    force_ids: list[str] | None = None,
) -> list[str]:
    """
    选取下一批可执行 task。

    @param plan plan.json
    @param max_parallel 并行上限
    @param force_ids 用户强制指定
    @return task id 列表
    """
    if force_ids:
        return pick_forced_tasks(plan, force_ids, max_parallel=max_parallel)

    done_ids = done_task_ids(plan)
    selected: list[str] = []

    for task in _task_list(plan):
        if len(selected) >= max_parallel:
            break
        status = str(task.get("status") or TaskStatus.PENDING)
        tid = str(task.get("id") or "")
        if not tid or tid in selected:
            continue
        if status == TaskStatus.FAILED:
            retry = int(task.get("retry_count") or 0)
            max_retries = int(task.get("max_retries") or 2)
            if retry <= max_retries:
                selected.append(tid)
        elif status == TaskStatus.PENDING and deps_satisfied(task, done_ids):
            selected.append(tid)

    return selected


def route_after_supervisor(state: dict[str, Any]) -> str:
    """
    Supervisor 之后的路由。

    @param state PlanState
    @return worker | synthesize | confirm | end
    """
    phase = str(state.get("phase") or "")
    if phase in (PlanPhase.CANCELLED, PlanPhase.COMPLETED):
        return "end"
    if state.get("cancel_requested"):
        return "end"

    batch = state.get("parallel_batch")
    if isinstance(batch, list) and batch:
        return "worker"

    plan = state.get("plan") if isinstance(state.get("plan"), dict) else {}
    if all_tasks_terminal(plan):
        return "synthesize"
    if has_blocking_failures(plan):
        return "confirm"
    if isinstance(batch, list) and not batch:
        return "confirm"
    return "end"


def supervisor_node(state: dict[str, Any], ctx: PlanRuntimeContext) -> dict[str, Any]:
    """
    Supervisor 节点：选择下一批 Worker task。

    @param state PlanState
    @param ctx 运行时上下文
    @return state 更新
    """
    if is_cancel_requested(ctx.thread_id) or state.get("cancel_requested"):
        snapshot = build_workflow_snapshot(
            thread_id=ctx.thread_id,
            phase=PlanPhase.CANCELLED,
            plan=state.get("plan") if isinstance(state.get("plan"), dict) else None,
            current_node="supervisor",
        )
        return {
            "phase": PlanPhase.CANCELLED,
            "parallel_batch": [],
            "cancel_requested": True,
            "workflow_snapshot": snapshot,
        }

    plan = dict(state.get("plan") or {})
    plan_id = str(plan.get("plan_id") or state.get("plan_id") or "")
    if plan_id:
        disk = load_plan(ctx.workspace, plan_id, plans_dir=ctx.settings.plans_dir)
        if disk:
            plan = disk

    plan, recovered = recover_stale_running_tasks(plan)
    if recovered and plan_id:
        save_plan(ctx.workspace, plan, plans_dir=ctx.settings.plans_dir)

    if has_blocking_failures(plan) and not all_tasks_terminal(plan):
        plan["phase"] = PlanPhase.AWAITING_CONFIRM
        if plan_id:
            save_plan(ctx.workspace, plan, plans_dir=ctx.settings.plans_dir)
        snapshot = build_workflow_snapshot(
            thread_id=ctx.thread_id,
            phase=PlanPhase.AWAITING_CONFIRM,
            plan=plan,
            current_node="confirm",
        )
        return {
            "plan": plan,
            "phase": PlanPhase.AWAITING_CONFIRM,
            "parallel_batch": [],
            "workflow_snapshot": snapshot,
        }

    force_raw = state.get("force_task_ids")
    force_ids = [str(x) for x in force_raw if str(x).strip()] if isinstance(force_raw, list) else None
    max_parallel = max(1, int(ctx.settings.max_parallel_workers or 1))
    batch = pick_next_tasks(plan, max_parallel=max_parallel, force_ids=force_ids)

    snapshot = build_workflow_snapshot(
        thread_id=ctx.thread_id,
        phase=PlanPhase.EXECUTING,
        plan=plan,
        current_node="supervisor",
        current_task_id=batch[0] if len(batch) == 1 else None,
    )
    update: dict[str, Any] = {
        "plan": plan,
        "phase": PlanPhase.EXECUTING,
        "parallel_batch": batch,
        "workflow_snapshot": snapshot,
        "force_task_ids": [],
    }
    if batch:
        update["current_task_id"] = batch[0]
    return update
