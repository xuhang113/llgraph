"""Agent / Plan 模式切换循环。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from llgraph.core.agent_session import AgentSessionContext
from llgraph.plan.config import resolve_plan_settings
from llgraph.session.mode_switch import SessionModeTransition
from llgraph.session.session_switch import switch_agent_thread
from llgraph.terminal.plan_session import build_plan_session_params, run_plan_session
from llgraph.terminal.session import TerminalSessionParams, run_terminal_session


@dataclass
class AgentPlanLoopContext:
    """Agent ↔ Plan 循环共享上下文。"""

    workspace: Path
    agent_session: AgentSessionContext
    terminal_params_factory: Callable[[], TerminalSessionParams]
    plan_common: dict[str, Any]
    session_hints: dict[str, str | None]


def run_agent_plan_loop(ctx: AgentPlanLoopContext) -> None:
    """
    在 Agent 与 Plan 模式间切换，直至用户 exit。

    @param ctx 循环上下文
    """
    settings = resolve_plan_settings(ctx.workspace)
    return_agent_thread: str | None = ctx.agent_session.thread_id
    plan_resume_hint = ""
    plan_thread: str | None = None
    plan_goal = ""

    while True:
        agent_transition = run_terminal_session(ctx.terminal_params_factory())
        if agent_transition is None:
            return
        if agent_transition.mode != "plan":
            return

        return_agent_thread = agent_transition.from_thread_id or return_agent_thread
        plan_thread = agent_transition.thread_id or f"plan-{uuid.uuid4().hex[:8]}"
        plan_goal = agent_transition.opening_goal
        plan_resume_hint = ""

        while True:
            plan_transition = run_plan_session(
                build_plan_session_params(
                    workspace=ctx.workspace,
                    thread_id=plan_thread,
                    settings=settings,
                    return_agent_thread_id=return_agent_thread,
                    source_agent_thread_id=return_agent_thread,
                    opening_goal=plan_goal,
                    resume_hint=plan_resume_hint,
                    **ctx.plan_common,
                )
            )
            plan_goal = ""
            plan_resume_hint = ""
            if plan_transition is None:
                return
            if plan_transition.mode == "agent":
                cli_tid = plan_transition.thread_id or return_agent_thread
                if cli_tid and cli_tid != ctx.agent_session.thread_id:
                    no_spill = False
                    spill = ctx.agent_session.context_spill
                    if spill is not None and bool(getattr(spill, "disabled", False)):
                        no_spill = True
                    switch_agent_thread(ctx.agent_session, cli_tid, no_spill=no_spill)
                ctx.session_hints["resume"] = "已从 Plan 模式切回 Agent 模式。"
                if plan_transition.handoff_report:
                    ctx.session_hints["resume"] += "（已保留 Plan 汇总，可继续提问）"
                break
            if plan_transition.mode == "plan":
                plan_thread = plan_transition.thread_id or plan_thread
                plan_goal = plan_transition.opening_goal
                plan_resume_hint = f"已切换到 {plan_thread}"
                continue
            return


def run_plan_agent_loop(
    *,
    workspace: Path,
    initial_plan_thread: str,
    opening_goal: str,
    plan_common: dict[str, Any],
    build_agent_session: Callable[[str], AgentSessionContext],
) -> None:
    """
    从 Plan 模式启动，支持切回 Agent。

    @param workspace 工作区
    @param initial_plan_thread 初始 plan thread
    @param opening_goal 计划目标
    @param plan_common PlanSessionParams 公共字段
    @param build_agent_session 按 cli thread 构建 AgentSessionContext
    """
    settings = resolve_plan_settings(workspace)
    plan_thread = initial_plan_thread
    plan_goal = opening_goal
    plan_resume_hint = plan_common.pop("resume_hint", "")
    return_agent_thread: str | None = None
    agent_session: AgentSessionContext | None = None

    while True:
        plan_transition = run_plan_session(
            build_plan_session_params(
                workspace=workspace,
                thread_id=plan_thread,
                settings=settings,
                opening_goal=plan_goal,
                resume_hint=plan_resume_hint,
                return_agent_thread_id=return_agent_thread,
                **plan_common,
            )
        )
        plan_goal = ""
        plan_resume_hint = ""
        if plan_transition is None:
            return
        if plan_transition.mode == "plan":
            plan_thread = plan_transition.thread_id or plan_thread
            plan_goal = plan_transition.opening_goal
            plan_resume_hint = f"已切换到 {plan_thread}"
            continue
        if plan_transition.mode != "agent":
            return

        cli_tid = plan_transition.thread_id or return_agent_thread or f"cli-{uuid.uuid4().hex[:8]}"
        if agent_session is None or agent_session.thread_id != cli_tid:
            agent_session = build_agent_session(cli_tid)
        return_agent_thread = cli_tid

        def _terminal_params() -> TerminalSessionParams:
            return TerminalSessionParams(
                agent=agent_session.agent,
                workspace=workspace,
                thread_id=agent_session.thread_id,
                trace_session=agent_session.trace_session,
                context_session=agent_session.context_session,
                allow_write=agent_session.allow_write,
                agent_session=agent_session,
                edit_tracker=agent_session.edit_tracker,
                write_failure_tracker=agent_session.write_failure_tracker,
                watch_active=plan_common.get("watch_active", False),
                web_search_enabled=agent_session.web_search_enabled,
                mcp_summary=plan_common.get("mcp_summary", ""),
                memory_kind=plan_common.get("memory_kind", "jsonl"),
            )

        agent_transition = run_terminal_session(_terminal_params())
        if agent_transition is None:
            return
        if agent_transition.mode != "plan":
            return
        plan_thread = agent_transition.thread_id or f"plan-{uuid.uuid4().hex[:8]}"
        plan_goal = agent_transition.opening_goal
        plan_resume_hint = "已从 Agent 模式切回 Plan 模式。"
