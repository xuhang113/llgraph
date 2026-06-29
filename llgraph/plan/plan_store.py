"""plan.json / result.json 读写。"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llgraph.plan.state import PlanPhase, TaskStatus


def plans_root(workspace: Path, plans_dir: str = ".llgraph/plans") -> Path:
    """
    Plan 落盘根目录。

    @param workspace 工作区根
    @param plans_dir 相对路径
    @return 绝对路径
    """
    rel = plans_dir.strip().lstrip("/") or ".llgraph/plans"
    return workspace.expanduser().resolve() / rel


def new_plan_id() -> str:
    """
    生成 plan_id。

    @return 8 位 hex
    """
    return uuid.uuid4().hex[:8]


def plan_dir(workspace: Path, plan_id: str, *, plans_dir: str = ".llgraph/plans") -> Path:
    """
    单个 Plan 目录。

    @param workspace 工作区根
    @param plan_id Plan ID
    @param plans_dir 相对路径
    @return Plan 目录
    """
    return plans_root(workspace, plans_dir) / plan_id


def plan_json_path(workspace: Path, plan_id: str, *, plans_dir: str = ".llgraph/plans") -> Path:
    """
    plan.json 路径。

    @param workspace 工作区根
    @param plan_id Plan ID
    @param plans_dir 相对路径
    @return plan.json 路径
    """
    return plan_dir(workspace, plan_id, plans_dir=plans_dir) / "plan.json"


def task_result_path(
    workspace: Path,
    plan_id: str,
    task_id: str,
    *,
    plans_dir: str = ".llgraph/plans",
) -> Path:
    """
    task result.json 路径。

    @param workspace 工作区根
    @param plan_id Plan ID
    @param task_id Task ID
    @param plans_dir 相对路径
    @return result.json 路径
    """
    return (
        plan_dir(workspace, plan_id, plans_dir=plans_dir)
        / "tasks"
        / task_id
        / "result.json"
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_placeholder_plan_title(title: str, plan_id: str = "") -> bool:
    """
    判断 plan.json 标题是否为占位（非用户/Planner 实质标题）。

    @param title plan title
    @param plan_id Plan ID
    @return 是否为占位标题
    """
    t = str(title or "").strip()
    if not t or t == "未命名计划":
        return True
    pid = str(plan_id or "").strip()
    if pid and t == f"Plan {pid}":
        return True
    return False


def empty_plan(*, plan_id: str, title: str = "", goal: str = "") -> dict[str, Any]:
    """
    创建空 plan 结构。

    @param plan_id Plan ID
    @param title 标题
    @param goal 目标说明
    @return plan dict
    """
    return {
        "plan_id": plan_id,
        "version": 1,
        "title": title or (f"Plan {plan_id}" if not goal.strip() else ""),
        "goal": goal,
        "phase": PlanPhase.PLANNING,
        "execution": {
            "allow_worker_write": False,
            "auto_run": True,
        },
        "tasks": [],
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }


def load_plan(workspace: Path, plan_id: str, *, plans_dir: str = ".llgraph/plans") -> dict[str, Any]:
    """
    读取 plan.json。

    @param workspace 工作区根
    @param plan_id Plan ID
    @param plans_dir 相对路径
    @return plan dict；不存在返回空 dict
    """
    path = plan_json_path(workspace, plan_id, plans_dir=plans_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def pick_richer_plan(
    plan_a: dict[str, Any] | None,
    plan_b: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    两份 plan 不一致时，优先任务更多的那份。

    @param plan_a 候选 A（如 plan.json）
    @param plan_b 候选 B（如 plan_state 内联）
    @return 合并结果
    """
    a = plan_a or {}
    b = plan_b or {}
    if not a:
        return b
    if not b:
        return a
    a_n = len(a.get("tasks") or []) if isinstance(a.get("tasks"), list) else 0
    b_n = len(b.get("tasks") or []) if isinstance(b.get("tasks"), list) else 0
    return b if b_n > a_n else a


def save_plan(
    workspace: Path,
    plan: dict[str, Any],
    *,
    plans_dir: str = ".llgraph/plans",
) -> Path:
    """
    写入 plan.json。

    @param workspace 工作区根
    @param plan plan dict
    @param plans_dir 相对路径
    @return 写入路径
    """
    plan_id = str(plan.get("plan_id") or new_plan_id())
    plan["plan_id"] = plan_id
    plan["updated_at"] = _now_iso()
    path = plan_json_path(workspace, plan_id, plans_dir=plans_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_task_result(
    workspace: Path,
    plan_id: str,
    task_id: str,
    result: dict[str, Any],
    *,
    plans_dir: str = ".llgraph/plans",
) -> Path:
    """
    写入 task result.json。

    @param workspace 工作区根
    @param plan_id Plan ID
    @param task_id Task ID
    @param result 结果 dict
    @param plans_dir 相对路径
    @return 写入路径
    """
    path = task_result_path(workspace, plan_id, task_id, plans_dir=plans_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = dict(result)
    result.setdefault("task_id", task_id)
    result.setdefault("updated_at", _now_iso())
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_task_result(
    workspace: Path,
    plan_id: str,
    task_id: str,
    *,
    plans_dir: str = ".llgraph/plans",
) -> dict[str, Any]:
    """
    读取 task result.json。

    @param workspace 工作区根
    @param plan_id Plan ID
    @param task_id Task ID
    @param plans_dir 相对路径
    @return result dict
    """
    path = task_result_path(workspace, plan_id, task_id, plans_dir=plans_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def update_task_status(
    plan: dict[str, Any],
    task_id: str,
    status: str,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    """
    更新 plan 内 task 状态。

    @param plan plan dict
    @param task_id Task ID
    @param status 新状态
    @param error 失败原因
    @return 更新后的 plan
    """
    tasks = plan.get("tasks")
    if not isinstance(tasks, list):
        return plan
    for task in tasks:
        if isinstance(task, dict) and str(task.get("id")) == task_id:
            task["status"] = status
            if error:
                task["error"] = error
            retry = int(task.get("retry_count") or 0)
            if status == TaskStatus.FAILED:
                task["retry_count"] = retry + 1
            break
    plan["updated_at"] = _now_iso()
    return plan


def count_task_progress(plan: dict[str, Any]) -> tuple[int, int]:
    """
    统计 task 进度。

    @param plan plan dict
    @return (done_count, total)
    """
    tasks = plan.get("tasks") if isinstance(plan.get("tasks"), list) else []
    total = len(tasks)
    done = sum(
        1
        for t in tasks
        if isinstance(t, dict) and t.get("status") in (TaskStatus.DONE, TaskStatus.SKIPPED)
    )
    return done, total
