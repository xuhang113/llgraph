"""Plan 模式配置（agent.json → plan 段）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llgraph.core.agent_config import load_agent_config


@dataclass(frozen=True)
class PlanSettings:
    """Plan 模式运行时配置。"""

    enabled: bool
    plans_dir: str
    planner_readonly: bool
    planner_max_turns: int
    worker_max_turns: int
    default_allow_write: bool
    max_parallel_workers: int
    confirm_via_survey: bool
    auto_run_after_confirm: bool
    step_confirm_each_task: bool
    handoff_enabled: bool
    plan_agent_context_enabled: bool
    agent_context_max_chars: int
    agent_context_max_messages: int
    auto_return_agent: bool


def resolve_plan_settings(workspace: Path) -> PlanSettings:
    """
    解析 Plan 配置。

    @param workspace 工作区根
    @return PlanSettings
    """
    cfg = load_agent_config(workspace)
    plan = cfg.get("plan") if isinstance(cfg.get("plan"), dict) else {}
    planner = plan.get("planner") if isinstance(plan.get("planner"), dict) else {}
    worker = plan.get("worker") if isinstance(plan.get("worker"), dict) else {}
    supervisor = plan.get("supervisor") if isinstance(plan.get("supervisor"), dict) else {}
    return PlanSettings(
        enabled=bool(plan.get("enabled", True)),
        plans_dir=str(plan.get("plans_dir") or ".llgraph/plans"),
        planner_readonly=bool(planner.get("readonly", True)),
        planner_max_turns=int(planner.get("max_turns") or 40),
        worker_max_turns=int(worker.get("max_turns") or 30),
        default_allow_write=bool(worker.get("default_allow_write", False)),
        max_parallel_workers=max(1, int(supervisor.get("max_parallel_workers") or 3)),
        confirm_via_survey=bool(plan.get("confirm_via_survey", True)),
        auto_run_after_confirm=bool(plan.get("auto_run_after_confirm", True)),
        step_confirm_each_task=bool(plan.get("step_confirm_each_task", False)),
        handoff_enabled=bool(plan.get("handoff_enabled", True)),
        plan_agent_context_enabled=bool(plan.get("plan_agent_context_enabled", True)),
        agent_context_max_chars=max(2000, int(plan.get("agent_context_max_chars") or 12000)),
        agent_context_max_messages=max(8, int(plan.get("agent_context_max_messages") or 48)),
        auto_return_agent=bool(plan.get("auto_return_agent", False)),
    )
