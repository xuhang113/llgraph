"""PlanGraph 父图状态定义。"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class PlanPhase:
    """Plan 阶段常量。"""

    PLANNING = "planning"
    AWAITING_CONFIRM = "awaiting_confirm"
    EXECUTING = "executing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskStatus:
    """Task 状态常量。"""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class NodeStatus:
    """Workflow node 状态。"""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    WAITING = "waiting"


class PlanState(TypedDict, total=False):
    """PlanGraph 父图 State。"""

    plan_id: str
    plan: dict[str, Any]
    phase: str
    current_task_id: str | None
    task_results: dict[str, dict[str, Any]]
    user_messages: Annotated[list[str], operator.add]
    final_report: str | None
    error: str | None
    workflow_snapshot: dict[str, Any]
    allow_worker_write: bool
    opening_goal: str
    source_agent_thread_id: str | None
    agent_context: str | None
    plan_version: int
    revision_note: str | None
    step_confirm_each_task: bool
    parallel_batch: list[str]
    cancel_requested: bool
