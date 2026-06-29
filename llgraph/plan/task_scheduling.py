"""Plan task 调度辅助：依赖校验、单 task 触发、僵死 running 恢复。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llgraph.plan.state import TaskStatus


def _task_list(plan: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = plan.get("tasks") if isinstance(plan.get("tasks"), list) else []
    return [t for t in tasks if isinstance(t, dict)]


def find_task(plan: dict[str, Any], task_id: str) -> dict[str, Any] | None:
    """按 id 查找 task。"""
    key = task_id.strip()
    for task in _task_list(plan):
        if str(task.get("id") or "") == key:
            return task
    return None


def done_task_ids(plan: dict[str, Any]) -> set[str]:
    """已完成（含 skipped）的 task id。"""
    return {
        str(t.get("id") or "")
        for t in _task_list(plan)
        if str(t.get("status") or "") in (TaskStatus.DONE, TaskStatus.SKIPPED)
    }


def all_work_task_ids(plan: dict[str, Any]) -> list[str]:
    """全部 work task id（保持 plan 顺序）。"""
    return [str(t.get("id") or "") for t in _task_list(plan) if str(t.get("id") or "")]


def _failed_retries_exhausted(task: dict[str, Any]) -> bool:
    status = str(task.get("status") or TaskStatus.PENDING)
    if status != TaskStatus.FAILED:
        return False
    retry = int(task.get("retry_count") or 0)
    max_retries = int(task.get("max_retries") or 2)
    return retry > max_retries


def deps_satisfied(task: dict[str, Any], done_ids: set[str]) -> bool:
    deps = task.get("depends_on")
    if not isinstance(deps, list) or not deps:
        return True
    return all(str(d) in done_ids for d in deps)


def missing_dependencies(plan: dict[str, Any], task_id: str) -> list[str]:
    """未满足的依赖 task id 列表。"""
    task = find_task(plan, task_id)
    if task is None:
        return []
    done_ids = done_task_ids(plan)
    deps = task.get("depends_on")
    if not isinstance(deps, list):
        return []
    return [str(d) for d in deps if str(d) not in done_ids]


def validate_task_runnable(
    plan: dict[str, Any],
    task_id: str,
    *,
    state: dict[str, Any] | None = None,
) -> tuple[bool, str, list[str]]:
    """
    判断指定 task 是否可立即执行。

    @param plan plan.json
    @param task_id 如 w1
    @param state 可选 PlanState，用于终止态门禁
    @return (可执行, 原因, 未满足依赖 id 列表)
    """
    if state is not None:
        from llgraph.plan.plan_lifecycle import worker_run_block_reason

        reason = worker_run_block_reason(state, action="run")
        if reason:
            return False, reason, []

    task = find_task(plan, task_id)
    if task is None:
        return False, f"Task 不存在: {task_id}", []

    status = str(task.get("status") or TaskStatus.PENDING)
    if status == TaskStatus.DONE:
        return False, f"{task_id} 已完成，点击可查看执行详情", []
    if status == TaskStatus.SKIPPED:
        return False, f"{task_id} 已跳过", []
    if status == TaskStatus.RUNNING:
        return False, f"{task_id} 正在执行中", []

    missing = missing_dependencies(plan, task_id)
    if missing:
        return (
            False,
            f"需先完成依赖: {', '.join(missing)}",
            missing,
        )

    if status == TaskStatus.FAILED and _failed_retries_exhausted(task):
        return False, f"{task_id} 已失败且重试次数用尽，请 /plan revise 调整计划", []

    if status not in (TaskStatus.PENDING, TaskStatus.FAILED):
        return False, f"{task_id} 当前状态为 {status}，无法执行", []

    return True, "", []


def reset_task_if_empty_write_done(
    workspace: Path,
    plan: dict[str, Any],
    task_id: str,
    *,
    plan_id: str,
    plans_dir: str,
) -> tuple[dict[str, Any], bool]:
    """
    可写 task 误标 done 且无 files_changed 时重置为 pending，便于重跑。

    @return (plan, 是否重置)
    """
    from llgraph.plan.plan_store import load_task_result, update_task_status

    task = find_task(plan, task_id)
    if task is None:
        return plan, False
    if bool(task.get("readonly")):
        return plan, False
    if str(task.get("status") or "") != TaskStatus.DONE:
        return plan, False
    result = load_task_result(workspace, plan_id, task_id, plans_dir=plans_dir)
    files = result.get("files_changed") if isinstance(result.get("files_changed"), list) else []
    if any(str(f).strip() for f in files):
        return plan, False
    plan = update_task_status(dict(plan), task_id, TaskStatus.PENDING, error=None)
    return plan, True


def pick_forced_tasks(
    plan: dict[str, Any],
    force_ids: list[str],
    *,
    max_parallel: int = 1,
) -> list[str]:
    """
    从用户指定的 task 中选取可执行项。

    @param plan plan.json
    @param force_ids 用户指定 id 列表
    @param max_parallel 并行上限
    @return 可执行的 task id（至多 max_parallel）
    """
    selected: list[str] = []
    for raw in force_ids:
        tid = str(raw or "").strip()
        if not tid or tid in selected:
            continue
        ok, _, _ = validate_task_runnable(plan, tid)
        if ok:
            selected.append(tid)
            if len(selected) >= max_parallel:
                break
    return selected


def recover_stale_running_tasks(plan: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """
    将无后台 job 时残留的 running 恢复为 pending（崩溃/中断后续跑）。

    @param plan plan.json
    @return (更新后的 plan, 是否有变更)
    """
    changed = False
    plan = dict(plan)
    tasks_out: list[Any] = []
    for task in _task_list(plan):
        row = dict(task)
        if str(row.get("status") or "") == TaskStatus.RUNNING:
            row["status"] = TaskStatus.PENDING
            changed = True
        tasks_out.append(row)
    if changed:
        plan["tasks"] = tasks_out
    return plan, changed


def plan_has_incomplete_tasks(plan: dict[str, Any]) -> bool:
    """是否仍有未终态 task（不含可自动重试的 failed）。"""
    for task in _task_list(plan):
        status = str(task.get("status") or TaskStatus.PENDING)
        if status in (TaskStatus.PENDING, TaskStatus.RUNNING):
            return True
        if status == TaskStatus.FAILED and not _failed_retries_exhausted(task):
            return True
    return False
