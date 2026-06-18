"""Plan 落盘并发控制（并行 Worker 安全更新 plan.json）。"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from llgraph.plan.plan_store import load_plan, save_plan

_lock_guard = threading.Lock()
_plan_locks: dict[str, threading.Lock] = {}


def plan_file_lock(workspace: Path, plan_id: str) -> threading.Lock:
    """
    获取 plan.json 文件级锁。

    @param workspace 工作区根
    @param plan_id Plan ID
    @return 互斥锁
    """
    key = f"{workspace.expanduser().resolve()}:{plan_id}"
    with _lock_guard:
        lock = _plan_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _plan_locks[key] = lock
        return lock


def mutate_plan_on_disk(
    workspace: Path,
    plan_id: str,
    mutator: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    plans_dir: str = ".llgraph/plans",
    fallback_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    在锁内读取 plan.json、变更并写回。

    @param workspace 工作区根
    @param plan_id Plan ID
    @param mutator 接收 plan dict 并返回更新后的 plan
    @param plans_dir 相对路径
    @param fallback_plan 磁盘无 plan 时的初始内容
    @return 写回后的 plan
    """
    lock = plan_file_lock(workspace, plan_id)
    with lock:
        plan = load_plan(workspace, plan_id, plans_dir=plans_dir)
        if not plan and fallback_plan:
            plan = dict(fallback_plan)
        if not plan:
            plan = dict(fallback_plan or {})
        updated = mutator(dict(plan))
        save_plan(workspace, updated, plans_dir=plans_dir)
        return updated
