"""从磁盘补齐 plan_state.task_results。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llgraph.plan.plan_store import load_task_result


def hydrate_task_results(
    workspace: Path,
    state: dict[str, Any],
    *,
    plans_dir: str,
) -> dict[str, Any]:
    """
    将 plan.json 各 task 的 result.json 合并进 state.task_results（内存缺失时）。

    @param workspace 工作区根
    @param state PlanState
    @param plans_dir plans 相对目录
    @return 更新后的 state 副本
    """
    merged = dict(state)
    plan = merged.get("plan") if isinstance(merged.get("plan"), dict) else {}
    plan_id = str(merged.get("plan_id") or plan.get("plan_id") or "")
    if not plan_id:
        return merged

    task_results = dict(merged.get("task_results") or {})
    tasks = plan.get("tasks") if isinstance(plan.get("tasks"), list) else []
    changed = False
    for task in tasks:
        if not isinstance(task, dict):
            continue
        tid = str(task.get("id") or "")
        if not tid or tid in task_results:
            continue
        disk = load_task_result(workspace, plan_id, tid, plans_dir=plans_dir)
        if disk:
            task_results[tid] = disk
            changed = True
    if changed:
        merged["task_results"] = task_results
    return merged
