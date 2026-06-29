"""Plan 会话服务。"""

from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.types import Command

from llgraph.config.edit_settings import resolve_edit_settings
from llgraph.core.write_failure_tracker import WriteFailureTracker
from llgraph.display.trace_display import TraceSession
from llgraph.plan.config import resolve_plan_settings
from llgraph.plan.execution_coordinator import is_running, job_snapshot, start_background, wait_until_done
from llgraph.plan.graph import build_plan_graph
from llgraph.plan.plan_registry import init_plan_session_meta
from llgraph.plan.plan_state_store import load_plan_state, save_plan_state
from llgraph.plan.plan_store import new_plan_id
from llgraph.plan.runner import (
    load_or_create_state,
    merge_plan_checkpoint_state,
    prepare_plan_for_resume,
    resume_after_confirm_decision,
    resume_executing_plan,
    run_until_interrupt,
)
from llgraph.plan.plan_discuss import run_plan_discuss
from llgraph.plan.plan_lifecycle import is_plan_closed, is_plan_terminal, needs_synthesize, worker_run_block_reason
from llgraph.plan.task_scheduling import validate_task_runnable, reset_task_if_empty_write_done
from llgraph.plan.runtime import PlanRuntimeContext
from llgraph.plan.state import PlanPhase, TaskStatus

from llgraph.console.runtime.event_hub import HUB
from llgraph.console.runtime.session_lock import LOCKS
from llgraph.console.runtime.sse_sink import SseTraceSink
from llgraph.console.runtime.workspace_runtime import RUNTIME_MANAGER


def create_plan_session(workspace: Path, *, goal: str = "") -> str:
    """
    创建新 Plan 会话目录与 meta。

    @param workspace 工作区根
    @param goal 初始目标（可选，后续 start 时传入）
    @return plan-* thread_id
    """
    thread_id = f"plan-{uuid.uuid4().hex[:8]}"
    plan_id = new_plan_id()
    init_plan_session_meta(
        workspace,
        thread_id,
        plan_id,
        phase=PlanPhase.PLANNING,
        title="",
    )
    goal_text = (goal or "").strip()
    if goal_text:
        from llgraph.session.session_meta import ensure_session_title_auto

        ensure_session_title_auto(workspace, thread_id, goal_text)
    return thread_id


def _resolve_plan_allow_write(workspace: Path, thread_id: str, cli_allow_write: bool) -> bool:
    """
    合并 confirm 写权限与工具栏 -w：execution.allow_worker_write 优先。

    @param workspace 工作区
    @param thread_id plan thread
    @param cli_allow_write 工具栏 -w
    @return 本次 Plan 运行是否允许写
    """
    state = load_plan_state(workspace, thread_id) or {}
    plan = state.get("plan") if isinstance(state.get("plan"), dict) else {}
    execution = plan.get("execution") if isinstance(plan.get("execution"), dict) else {}
    if execution.get("allow_worker_write") or state.get("allow_worker_write"):
        return True
    return cli_allow_write


def _build_plan_ctx(workspace: Path, thread_id: str, *, allow_write: bool) -> PlanRuntimeContext:
    effective_write = _resolve_plan_allow_write(workspace, thread_id, allow_write)
    settings = resolve_plan_settings(workspace)
    rt = RUNTIME_MANAGER.get(workspace, allow_write=effective_write)
    edit_settings = resolve_edit_settings(workspace)
    write_failure_tracker = (
        WriteFailureTracker(
            rt.context_session,
            failures_before_hint=edit_settings.write_failures_before_hint,
            chunk_max_chars=edit_settings.write_chunk_max_chars,
        )
        if effective_write
        else None
    )
    trace = TraceSession(mode=rt.trace_session.mode)
    return PlanRuntimeContext(
        workspace=workspace,
        thread_id=thread_id,
        settings=settings,
        trace_session=trace,
        context_session=rt.context_session,
        allow_write_cli=effective_write,
        mcp_tools=list(rt.mcp_tools),
        sandbox_policy=rt.sandbox_policy,
        web_search_enabled=rt.web_search_enabled,
        write_failure_tracker=write_failure_tracker,
    )


def _attach_sink(
    ctx: PlanRuntimeContext,
    channel: str,
    loop: asyncio.AbstractEventLoop,
    *,
    workspace: Path,
    thread_id: str,
) -> None:
    def emit(event: dict[str, Any]) -> None:
        HUB.publish_sync(channel, event, loop)

    ctx.sse_emit = emit
    ctx.sse_loop = loop
    from llgraph.console.runtime.sse_sink import PersistingSseTraceSink, SseTraceSink

    inner = SseTraceSink(emit)
    ctx.trace_session.trace_sink = PersistingSseTraceSink(
        inner,
        workspace=workspace,
        thread_id=thread_id,
    )


def _persist_web_trace(ctx: PlanRuntimeContext, workspace: Path, thread_id: str) -> None:
    """
    将 Plan 本轮 trace 落盘，供 Web 切换会话后从 lastTrace API 恢复。

    @param ctx Plan 运行时上下文
    @param workspace 工作区根
    @param thread_id plan thread
    """
    trace = ctx.trace_session
    sink = trace.trace_sink
    log_lines = list(getattr(sink, "log_lines", []) or [])
    step_payloads: list[dict[str, Any]] = []
    if trace.last_turn_steps:
        from llgraph.console.runtime.sse_sink import _step_to_dict

        step_payloads = [_step_to_dict(s) for s in trace.last_turn_steps]
    if not log_lines and not step_payloads:
        return
    from llgraph.session.web_trace_store import save_last_web_trace

    save_last_web_trace(
        workspace,
        thread_id,
        log_lines=log_lines,
        steps=step_payloads,
    )


def _append_confirm_history(
    workspace: Path,
    thread_id: str,
    state: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    """
    记录 Plan 确认/修订决策，供 Web 回看。

    @param workspace 工作区根
    @param thread_id plan thread
    @param state 当前 PlanState
    @param decision 用户确认载荷
    @return 更新后的 state
    """
    plan = state.get("plan") if isinstance(state.get("plan"), dict) else {}
    tasks = plan.get("tasks") if isinstance(plan.get("tasks"), list) else []
    entry = {
        "at": datetime.now(timezone.utc).isoformat(),
        "action": str(decision.get("action") or "approve").strip().lower(),
        "allow_worker_write": bool(decision.get("allow_worker_write")),
        "revise_note": str(decision.get("revise_note") or "").strip(),
        "plan_version": int(state.get("plan_version") or plan.get("version") or 1),
        "title": str(plan.get("title") or ""),
        "task_count": len(tasks),
        "tasks": [
            {"id": str(t.get("id") or ""), "title": str(t.get("title") or "")}
            for t in tasks
            if isinstance(t, dict)
        ],
    }
    history = state.get("confirm_history")
    if not isinstance(history, list):
        history = []
    history = [*history, entry][-20:]
    merged = {**state, "confirm_history": history}
    save_plan_state(workspace, thread_id, merged)
    return merged


def _emit_state(channel: str, loop: asyncio.AbstractEventLoop, workspace: Path, thread_id: str) -> None:
    from llgraph.console.discovery import load_plan_detail

    detail = load_plan_detail(workspace, thread_id)
    HUB.publish_sync(
        channel,
        {
            "type": "plan_state",
            "phase": detail.get("phase"),
            "workflow_snapshot": detail.get("workflow_snapshot"),
            "tasks_done": sum(
                1
                for t in detail.get("tasks") or []
                if isinstance(t, dict) and t.get("status") == "done"
            ),
            "tasks_total": len(detail.get("tasks") or []),
            "error": detail.get("error"),
            "final_report": detail.get("final_report"),
        },
        loop,
    )
    intr = detail.get("plan_state", {}).get("pending_interrupt")
    job = job_snapshot(thread_id)
    HUB.publish_sync(
        channel,
        {
            "type": "plan_job",
            "running": job.get("running", False),
            "error": job.get("error"),
        },
        loop,
    )


def _run_plan_graph(
    workspace: Path,
    thread_id: str,
    *,
    allow_write: bool,
    channel: str,
    loop: asyncio.AbstractEventLoop,
    input_payload: Any | None = None,
    opening_goal: str = "",
    state: dict[str, Any] | None = None,
    block: bool = False,
) -> None:
    ok, holder = LOCKS.acquire(thread_id, owner="web")
    if not ok and holder is not None:
        HUB.publish_sync(
            channel,
            {"type": "error", "message": f"Plan {thread_id} 被 {holder.owner} 占用"},
            loop,
        )
        HUB.publish_sync(channel, {"type": "end"}, loop)
        return

    ctx = _build_plan_ctx(workspace, thread_id, allow_write=allow_write)
    _attach_sink(ctx, channel, loop, workspace=workspace, thread_id=thread_id)
    graph = build_plan_graph(ctx)

    def _runner() -> None:
        try:
            HUB.publish_sync(
                channel,
                {"type": "plan_job", "running": True, "thread_id": thread_id},
                loop,
            )
            HUB.publish_sync(channel, {"type": "turn_start", "thread_id": thread_id}, loop)
            run_state = state
            if run_state is None:
                run_state, _ = load_or_create_state(
                    ctx,
                    graph,
                    opening_goal=opening_goal,
                )
            new_state, intr = run_until_interrupt(
                ctx,
                graph,
                run_state,
                input_payload=input_payload,
            )
            _emit_state(channel, loop, workspace, thread_id)
            if intr is not None:
                HUB.publish_sync(channel, {"type": "interrupt", "payload": intr}, loop)
            else:
                HUB.publish_sync(channel, {"type": "plan_done"}, loop)
        except Exception as exc:
            HUB.publish_sync(channel, {"type": "error", "message": str(exc)}, loop)
        finally:
            _persist_web_trace(ctx, workspace, thread_id)
            LOCKS.release(thread_id, owner="web")
            HUB.publish_sync(
                channel,
                {"type": "plan_job", "running": False, "thread_id": thread_id},
                loop,
            )
            HUB.publish_sync(channel, {"type": "end"}, loop)

    if block:
        _runner()
    else:
        started = start_background(thread_id, _runner)
        if not started:
            LOCKS.release(thread_id, owner="web")
            HUB.publish_sync(channel, {"type": "error", "message": "Plan 已在运行中"}, loop)
            HUB.publish_sync(channel, {"type": "end"}, loop)


def start_plan_with_goal(
    workspace: Path,
    thread_id: str,
    goal: str,
    *,
    allow_write: bool,
    channel: str,
    loop: asyncio.AbstractEventLoop,
    block_first: bool = True,
) -> None:
    """
    以 goal 启动 Plan 图。

    @param workspace 工作区
    @param thread_id plan thread
    @param goal 用户目标
    @param allow_write 是否允许写
    @param channel SSE channel
    @param loop 事件循环
    @param block_first Planner 首次是否阻塞
    """
    ctx = _build_plan_ctx(workspace, thread_id, allow_write=allow_write)
    graph = build_plan_graph(ctx)
    state, is_new = load_or_create_state(ctx, graph, opening_goal=goal)
    if not is_new and is_plan_closed(state) and goal.strip():
        HUB.publish_sync(
            channel,
            {
                "type": "error",
                "message": "Plan 已取消或已完成，请新建 Plan 或使用 /plan revise。",
            },
            loop,
        )
        HUB.publish_sync(channel, {"type": "end"}, loop)
        return
    if not is_new and is_plan_terminal(state) and goal.strip():
        HUB.publish_sync(
            channel,
            {
                "type": "error",
                "message": worker_run_block_reason(state, action="run")
                + " 请使用追问或 /plan revise。",
            },
            loop,
        )
        HUB.publish_sync(channel, {"type": "end"}, loop)
        return
    if is_new:
        init_plan_session_meta(
            workspace,
            thread_id,
            str(state.get("plan_id") or new_plan_id()),
            phase=str(state.get("phase") or PlanPhase.PLANNING),
        )
    state["opening_goal"] = goal.strip()

    goal_text = goal.strip()
    if goal_text:
        from llgraph.session.session_meta import ensure_session_title_auto

        ensure_session_title_auto(workspace, thread_id, goal_text)

    def _runner() -> None:
        _run_plan_graph(
            workspace,
            thread_id,
            allow_write=allow_write,
            channel=channel,
            loop=loop,
            state=state,
            block=block_first,
        )

    if block_first:
        threading.Thread(target=_runner, daemon=True).start()
    else:
        _runner()


def confirm_plan(
    workspace: Path,
    thread_id: str,
    decision: dict[str, Any],
    *,
    allow_write: bool,
    channel: str,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """
    Plan 确认后继续。

    @param workspace 工作区
    @param thread_id plan thread
    @param decision 确认决策
    @param allow_write 是否允许写
    @param channel SSE channel
    @param loop 事件循环
    """

    def _bg() -> None:
        ok, holder = LOCKS.acquire(thread_id, owner="web")
        if not ok and holder is not None:
            HUB.publish_sync(channel, {"type": "error", "message": "会话被占用"}, loop)
            HUB.publish_sync(channel, {"type": "end"}, loop)
            return
        ctx = _build_plan_ctx(workspace, thread_id, allow_write=allow_write)
        _attach_sink(ctx, channel, loop, workspace=workspace, thread_id=thread_id)
        graph = build_plan_graph(ctx)
        state = merge_plan_checkpoint_state(ctx, graph, load_plan_state(workspace, thread_id) or {})
        try:
            HUB.publish_sync(
                channel,
                {"type": "plan_job", "running": True, "thread_id": thread_id},
                loop,
            )
            HUB.publish_sync(channel, {"type": "turn_start", "thread_id": thread_id}, loop)
            state = _append_confirm_history(workspace, thread_id, state, decision)
            new_state, intr = resume_after_confirm_decision(ctx, graph, state, decision)
            _emit_state(channel, loop, workspace, thread_id)
            if intr is not None:
                HUB.publish_sync(channel, {"type": "interrupt", "payload": intr}, loop)
            else:
                HUB.publish_sync(channel, {"type": "plan_done"}, loop)
        except Exception as exc:
            HUB.publish_sync(channel, {"type": "error", "message": str(exc)}, loop)
        finally:
            _persist_web_trace(ctx, workspace, thread_id)
            LOCKS.release(thread_id, owner="web")
            HUB.publish_sync(
                channel,
                {"type": "plan_job", "running": False, "thread_id": thread_id},
                loop,
            )
            HUB.publish_sync(channel, {"type": "end"}, loop)

    threading.Thread(target=_bg, daemon=True).start()


def continue_plan(
    workspace: Path,
    thread_id: str,
    *,
    allow_write: bool,
    channel: str,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """
    Plan 继续执行（task_step_confirm 或 executing 恢复）。

    @param workspace 工作区
    @param thread_id plan thread
    @param allow_write 是否允许写
    @param channel SSE channel
    @param loop 事件循环
    """

    def _bg() -> None:
        ok, holder = LOCKS.acquire(thread_id, owner="web")
        if not ok and holder is not None:
            HUB.publish_sync(channel, {"type": "error", "message": "会话被占用"}, loop)
            HUB.publish_sync(channel, {"type": "end"}, loop)
            return
        ctx = _build_plan_ctx(workspace, thread_id, allow_write=allow_write)
        _attach_sink(ctx, channel, loop, workspace=workspace, thread_id=thread_id)
        graph = build_plan_graph(ctx)
        state = merge_plan_checkpoint_state(ctx, graph, load_plan_state(workspace, thread_id) or {})
        try:
            HUB.publish_sync(
                channel,
                {"type": "plan_job", "running": True, "thread_id": thread_id},
                loop,
            )
            HUB.publish_sync(channel, {"type": "turn_start", "thread_id": thread_id}, loop)
            state = prepare_plan_for_resume(ctx, state)
            if is_plan_terminal(state):
                HUB.publish_sync(
                    channel,
                    {"type": "error", "message": worker_run_block_reason(state, action="continue")},
                    loop,
                )
                return
            block = worker_run_block_reason(state, action="continue")
            if block and not needs_synthesize(state):
                HUB.publish_sync(channel, {"type": "error", "message": block}, loop)
                return
            save_plan_state(workspace, thread_id, state)
            config = {"configurable": {"thread_id": thread_id}}
            from llgraph.plan.plan_resume import graph_has_interrupt

            if graph_has_interrupt(graph, config):
                new_state, intr = run_until_interrupt(
                    ctx,
                    graph,
                    state,
                    input_payload=Command(resume={}),
                )
            else:
                pending = state.get("pending_interrupt")
                if isinstance(pending, dict) and pending.get("type") == "plan_confirm":
                    HUB.publish_sync(
                        channel,
                        {"type": "interrupt", "payload": pending},
                        loop,
                    )
                    _emit_state(channel, loop, workspace, thread_id)
                    return
                new_state, intr = resume_executing_plan(ctx, graph, state)
            _emit_state(channel, loop, workspace, thread_id)
            if intr is not None:
                HUB.publish_sync(channel, {"type": "interrupt", "payload": intr}, loop)
            else:
                HUB.publish_sync(channel, {"type": "plan_done"}, loop)
        except Exception as exc:
            HUB.publish_sync(channel, {"type": "error", "message": str(exc)}, loop)
        finally:
            _persist_web_trace(ctx, workspace, thread_id)
            LOCKS.release(thread_id, owner="web")
            HUB.publish_sync(
                channel,
                {"type": "plan_job", "running": False, "thread_id": thread_id},
                loop,
            )
            HUB.publish_sync(channel, {"type": "end"}, loop)

    if is_running(thread_id):
        HUB.publish_sync(channel, {"type": "error", "message": "Plan 已在运行"}, loop)
        HUB.publish_sync(channel, {"type": "end"}, loop)
        return
    threading.Thread(target=_bg, daemon=True).start()


def run_plan_task(
    workspace: Path,
    thread_id: str,
    task_id: str,
    *,
    allow_write: bool,
    channel: str,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """
    手动执行单个 Work task（依赖未满足时由调用方校验）。

    @param workspace 工作区
    @param thread_id plan thread
    @param task_id 如 w1
    @param allow_write 是否允许写
    @param channel SSE channel
    @param loop 事件循环
    """

    def _bg() -> None:
        ok, holder = LOCKS.acquire(thread_id, owner="web")
        if not ok and holder is not None:
            HUB.publish_sync(channel, {"type": "error", "message": "会话被占用"}, loop)
            HUB.publish_sync(channel, {"type": "end"}, loop)
            return
        ctx = _build_plan_ctx(workspace, thread_id, allow_write=allow_write)
        _attach_sink(ctx, channel, loop, workspace=workspace, thread_id=thread_id)
        graph = build_plan_graph(ctx)
        state = merge_plan_checkpoint_state(ctx, graph, load_plan_state(workspace, thread_id) or {})
        try:
            HUB.publish_sync(
                channel,
                {"type": "plan_job", "running": True, "thread_id": thread_id},
                loop,
            )
            HUB.publish_sync(channel, {"type": "turn_start", "thread_id": thread_id}, loop)
            state = prepare_plan_for_resume(ctx, state)
            plan = state.get("plan") if isinstance(state.get("plan"), dict) else {}
            plan_id = str(plan.get("plan_id") or state.get("plan_id") or "")
            if plan_id:
                plan, reset = reset_task_if_empty_write_done(
                    workspace,
                    plan,
                    task_id,
                    plan_id=plan_id,
                    plans_dir=ctx.settings.plans_dir,
                )
                if reset:
                    from llgraph.plan.plan_store import save_plan

                    save_plan(workspace, plan, plans_dir=ctx.settings.plans_dir)
                    state["plan"] = plan
            block = worker_run_block_reason(state, action="run")
            if block:
                HUB.publish_sync(channel, {"type": "error", "message": block}, loop)
                return
            runnable, message, _ = validate_task_runnable(plan, task_id, state=state)
            if not runnable:
                HUB.publish_sync(channel, {"type": "error", "message": message}, loop)
                return
            state["force_task_ids"] = [task_id]
            if state.get("phase") != PlanPhase.EXECUTING:
                state["phase"] = PlanPhase.EXECUTING
            save_plan_state(workspace, thread_id, state)
            new_state, intr = resume_executing_plan(ctx, graph, state)
            _emit_state(channel, loop, workspace, thread_id)
            if intr is not None:
                HUB.publish_sync(channel, {"type": "interrupt", "payload": intr}, loop)
            else:
                HUB.publish_sync(channel, {"type": "plan_done"}, loop)
        except Exception as exc:
            HUB.publish_sync(channel, {"type": "error", "message": str(exc)}, loop)
        finally:
            _persist_web_trace(ctx, workspace, thread_id)
            LOCKS.release(thread_id, owner="web")
            HUB.publish_sync(
                channel,
                {"type": "plan_job", "running": False, "thread_id": thread_id},
                loop,
            )
            HUB.publish_sync(channel, {"type": "end"}, loop)

    if is_running(thread_id):
        HUB.publish_sync(channel, {"type": "error", "message": "Plan 已在运行"}, loop)
        HUB.publish_sync(channel, {"type": "end"}, loop)
        return
    threading.Thread(target=_bg, daemon=True).start()


def check_plan_task_runnable(workspace: Path, thread_id: str, task_id: str) -> dict[str, Any]:
    """
    检查 Work task 是否可执行。

    @param workspace 工作区
    @param thread_id plan thread
    @param task_id 如 w1
    @return ok / message / missing_deps
    """
    ctx = _build_plan_ctx(workspace, thread_id, allow_write=False)
    state = load_plan_state(workspace, thread_id) or {}
    state = prepare_plan_for_resume(ctx, state)
    plan = state.get("plan") if isinstance(state.get("plan"), dict) else {}
    ok, message, missing = validate_task_runnable(plan, task_id, state=state)
    return {"ok": ok, "message": message, "missing_deps": missing}


def discuss_plan(
    workspace: Path,
    thread_id: str,
    message: str,
    *,
    channel: str,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """
    Plan 终止后只读问答（SSE turn_done）。

    @param workspace 工作区
    @param thread_id plan thread
    @param message 用户问题
    @param channel SSE channel
    @param loop 事件循环
    """

    def _bg() -> None:
        ok, holder = LOCKS.acquire(thread_id, owner="web")
        if not ok and holder is not None:
            HUB.publish_sync(channel, {"type": "error", "message": "会话被占用"}, loop)
            HUB.publish_sync(channel, {"type": "end"}, loop)
            return
        ctx = _build_plan_ctx(workspace, thread_id, allow_write=False)
        _attach_sink(ctx, channel, loop, workspace=workspace, thread_id=thread_id)
        state = load_plan_state(workspace, thread_id) or {}
        try:
            if message.strip():
                from llgraph.session.session_meta import ensure_session_title_auto

                ensure_session_title_auto(workspace, thread_id, message.strip())
            HUB.publish_sync(channel, {"type": "turn_start", "thread_id": thread_id}, loop)
            text = run_plan_discuss(ctx, state, message)
            from llgraph.context.message_normalize import format_agent_chat_display_text

            display = format_agent_chat_display_text(text)
            HUB.publish_sync(
                channel,
                {"type": "turn_done", "text": display, "thread_id": thread_id},
                loop,
            )
        except Exception as exc:
            HUB.publish_sync(channel, {"type": "error", "message": str(exc)}, loop)
        finally:
            _persist_web_trace(ctx, workspace, thread_id)
            LOCKS.release(thread_id, owner="web")
            HUB.publish_sync(channel, {"type": "end"}, loop)

    threading.Thread(target=_bg, daemon=True).start()


def get_plan_status(thread_id: str) -> dict[str, Any]:
    """
    获取 Plan job 状态。

    @param thread_id plan thread
    @return job snapshot
    """
    return job_snapshot(thread_id)


def _force_stop_running_plan(workspace: Path, thread_id: str) -> dict[str, Any]:
    """
    立即停止 Plan：标记 cancel、跳过所有未完成任务、清空 parallel batch。

    @param workspace 工作区
    @param thread_id plan thread
    @return 更新后的 plan_state 片段
    """
    from llgraph.plan.execution_coordinator import request_cancel, request_cancel_all_tasks
    from llgraph.plan.plan_state_store import load_plan_state, save_plan_state
    from llgraph.plan.plan_store import load_plan, save_plan
    from llgraph.plan.workflow_view import build_workflow_snapshot

    request_cancel(thread_id)
    ctx = _build_plan_ctx(workspace, thread_id, allow_write=False)
    state = load_plan_state(workspace, thread_id) or {}
    plan = dict(state.get("plan") or {})
    plan_id = str(plan.get("plan_id") or state.get("plan_id") or "")
    settings = ctx.settings
    if plan_id:
        disk = load_plan(ctx.workspace, plan_id, plans_dir=settings.plans_dir)
        if disk:
            plan = dict(disk)

    skip_ids: list[str] = []
    raw_tasks = plan.get("tasks") if isinstance(plan.get("tasks"), list) else []
    tasks_out: list[Any] = []
    for task in raw_tasks:
        if not isinstance(task, dict):
            continue
        row = dict(task)
        tid = str(row.get("id") or "").strip()
        status = str(row.get("status") or TaskStatus.PENDING)
        if status not in (TaskStatus.DONE, TaskStatus.SKIPPED):
            row["status"] = TaskStatus.SKIPPED
            if not str(row.get("error") or "").strip():
                row["error"] = "用户停止"
            if tid:
                skip_ids.append(tid)
        tasks_out.append(row)
    if skip_ids:
        request_cancel_all_tasks(thread_id, skip_ids)

    plan["tasks"] = tasks_out
    if plan_id:
        save_plan(ctx.workspace, plan, plans_dir=settings.plans_dir)

    phase = str(state.get("phase") or "")
    next_phase = PlanPhase.CANCELLED

    snapshot = build_workflow_snapshot(
        thread_id=thread_id,
        phase=next_phase,
        plan=plan,
        current_node="supervisor" if phase == PlanPhase.EXECUTING else "planner",
    )
    state.update(
        {
            "plan": plan,
            "phase": next_phase,
            "cancel_requested": True,
            "parallel_batch": [],
            "workflow_snapshot": snapshot,
        }
    )
    save_plan_state(workspace, thread_id, state)
    if plan_id:
        init_plan_session_meta(
            ctx.workspace,
            thread_id,
            plan_id,
            phase=PlanPhase.CANCELLED,
            title=str(plan.get("title") or ""),
        )
    return state


def cancel_plan(workspace: Path, thread_id: str) -> dict[str, Any]:
    """
    立即停止 Plan 后台执行（跳过所有未完成 Work，不再调度新 batch）。

    @param workspace 工作区
    @param thread_id plan thread
    @return ok / message
    """
    from llgraph.plan.execution_coordinator import is_running

    running = is_running(thread_id)
    if running:
        _force_stop_running_plan(workspace, thread_id)
        return {"ok": True, "message": ""}
    state = load_plan_state(workspace, thread_id) or {}
    phase = str(state.get("phase") or "")
    if phase in (
        PlanPhase.PLANNING,
        PlanPhase.EXECUTING,
        PlanPhase.AWAITING_CONFIRM,
    ):
        _force_stop_running_plan(workspace, thread_id)
        return {"ok": True, "message": "Plan 已停止"}
    return {"ok": False, "message": "Plan 当前未在运行"}


def abort_plan(workspace: Path, thread_id: str) -> dict[str, Any]:
    """
    取消 Plan（运行中则先 request_cancel，并将未完成 task 标为 skipped、phase 置 cancelled）。

    @param workspace 工作区
    @param thread_id plan thread
    @return ok / message
    """
    from llgraph.plan.execution_coordinator import (
        clear_cancelled_tasks,
        is_running,
        request_cancel,
    )
    from llgraph.plan.plan_registry import init_plan_session_meta
    from llgraph.plan.plan_state_store import load_plan_state, save_plan_state
    from llgraph.plan.plan_store import load_plan, save_plan
    from llgraph.plan.runner import prepare_plan_for_resume
    from llgraph.plan.workflow_view import build_workflow_snapshot

    running = is_running(thread_id)
    if running:
        request_cancel(thread_id)

    ctx = _build_plan_ctx(workspace, thread_id, allow_write=False)
    state = prepare_plan_for_resume(ctx, load_plan_state(workspace, thread_id) or {})
    plan = dict(state.get("plan") or {})
    plan_id = str(plan.get("plan_id") or state.get("plan_id") or "")
    if plan_id:
        disk = load_plan(ctx.workspace, plan_id, plans_dir=ctx.settings.plans_dir)
        if disk:
            plan = disk

    raw_tasks = plan.get("tasks") if isinstance(plan.get("tasks"), list) else []
    tasks_out: list[Any] = []
    for task in raw_tasks:
        if not isinstance(task, dict):
            continue
        row = dict(task)
        status = str(row.get("status") or TaskStatus.PENDING)
        if status not in (TaskStatus.DONE, TaskStatus.SKIPPED):
            row["status"] = TaskStatus.SKIPPED
            if not str(row.get("error") or "").strip():
                row["error"] = "Plan 已取消"
        tasks_out.append(row)
    plan["tasks"] = tasks_out
    plan["phase"] = PlanPhase.CANCELLED

    if plan_id:
        save_plan(ctx.workspace, plan, plans_dir=ctx.settings.plans_dir)
        init_plan_session_meta(
            ctx.workspace,
            thread_id,
            plan_id,
            phase=PlanPhase.CANCELLED,
            title=str(plan.get("title") or ""),
        )

    snapshot = build_workflow_snapshot(
        thread_id=thread_id,
        phase=PlanPhase.CANCELLED,
        plan=plan,
        current_node="supervisor",
    )
    state.update(
        {
            "plan": plan,
            "phase": PlanPhase.CANCELLED,
            "cancel_requested": True,
            "parallel_batch": [],
            "workflow_snapshot": snapshot,
        }
    )
    save_plan_state(workspace, thread_id, state)
    clear_cancelled_tasks(thread_id)

    if not running:
        from llgraph.console.runtime.session_lock import release_stale_web_lock

        release_stale_web_lock(thread_id)
        return {"ok": True, "message": "Plan 已取消，可删除会话或新建 Plan"}

    return {
        "ok": True,
        "message": "已请求停止；未完成 Work 已标记跳过，后台线程结束后可删除会话",
    }


def cancel_plan_task(workspace: Path, thread_id: str, task_id: str) -> dict[str, Any]:
    """
    停止/跳过单个 Work task。

    @param workspace 工作区
    @param thread_id plan thread
    @param task_id 如 w1
    @return ok / message
    """
    from llgraph.plan.execution_coordinator import (
        is_running,
        request_cancel,
        request_cancel_task,
    )
    from llgraph.plan.plan_state_store import load_plan_state, save_plan_state
    from llgraph.plan.plan_store import load_plan, save_plan, update_task_status
    from llgraph.plan.task_scheduling import find_task

    tid = (task_id or "").strip()
    if not tid:
        return {"ok": False, "message": "task_id 无效"}

    state = load_plan_state(workspace, thread_id) or {}
    plan = state.get("plan") if isinstance(state.get("plan"), dict) else {}
    plan_id = str(state.get("plan_id") or plan.get("plan_id") or "")
    settings = resolve_plan_settings(workspace)
    if plan_id:
        disk = load_plan(workspace, plan_id, plans_dir=settings.plans_dir)
        if disk:
            plan = disk

    task = find_task(plan, tid)
    if task is None:
        return {"ok": False, "message": f"Task 不存在: {tid}"}

    status = str(task.get("status") or TaskStatus.PENDING)
    if status in (TaskStatus.DONE, TaskStatus.SKIPPED):
        return {"ok": False, "message": f"{tid} 已终态 ({status})"}

    request_cancel_task(thread_id, tid)
    if is_running(thread_id):
        request_cancel(thread_id)

    plan = update_task_status(dict(plan), tid, TaskStatus.SKIPPED, error="用户取消")
    if plan_id:
        save_plan(workspace, plan, plans_dir=settings.plans_dir)
    state["plan"] = plan
    save_plan_state(workspace, thread_id, state)
    return {"ok": True, "message": f"已停止 {tid}"}


def wait_plan_done(thread_id: str, timeout: float = 3600.0) -> bool:
    """
    等待 Plan 后台任务结束。

    @param thread_id plan thread
    @param timeout 超时秒
    @return 是否在超时前结束
    """
    return wait_until_done(thread_id, timeout=timeout)
