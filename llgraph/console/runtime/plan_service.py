"""Plan 会话服务。"""

from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import dataclass
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
    prepare_plan_for_resume,
    resume_after_confirm_decision,
    resume_executing_plan,
    run_until_interrupt,
)
from llgraph.plan.plan_discuss import run_plan_discuss
from llgraph.plan.plan_lifecycle import is_plan_terminal, needs_synthesize, worker_run_block_reason
from llgraph.plan.task_scheduling import validate_task_runnable
from llgraph.plan.runtime import PlanRuntimeContext
from llgraph.plan.state import PlanPhase

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
    return thread_id


def _build_plan_ctx(workspace: Path, thread_id: str, *, allow_write: bool) -> PlanRuntimeContext:
    settings = resolve_plan_settings(workspace)
    rt = RUNTIME_MANAGER.get(workspace, allow_write=allow_write)
    edit_settings = resolve_edit_settings(workspace)
    write_failure_tracker = (
        WriteFailureTracker(
            rt.context_session,
            failures_before_hint=edit_settings.write_failures_before_hint,
            chunk_max_chars=edit_settings.write_chunk_max_chars,
        )
        if allow_write
        else None
    )
    trace = TraceSession(mode=rt.trace_session.mode)
    return PlanRuntimeContext(
        workspace=workspace,
        thread_id=thread_id,
        settings=settings,
        trace_session=trace,
        context_session=rt.context_session,
        allow_write_cli=allow_write,
        mcp_tools=list(rt.mcp_tools),
        sandbox_policy=rt.sandbox_policy,
        web_search_enabled=rt.web_search_enabled,
        write_failure_tracker=write_failure_tracker,
    )


def _attach_sink(ctx: PlanRuntimeContext, channel: str, loop: asyncio.AbstractEventLoop) -> None:
    def emit(event: dict[str, Any]) -> None:
        HUB.publish_sync(channel, event, loop)

    ctx.trace_session.trace_sink = SseTraceSink(emit)


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
        return

    ctx = _build_plan_ctx(workspace, thread_id, allow_write=allow_write)
    _attach_sink(ctx, channel, loop)
    graph = build_plan_graph(ctx)

    def _runner() -> None:
        try:
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
            LOCKS.release(thread_id, owner="web")
            HUB.publish_sync(channel, {"type": "end"}, loop)

    if block:
        _runner()
    else:
        started = start_background(thread_id, _runner)
        if not started:
            HUB.publish_sync(channel, {"type": "error", "message": "Plan 已在运行中"}, loop)


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

    def _bg() -> None:
        _run_plan_graph(
            workspace,
            thread_id,
            allow_write=allow_write,
            channel=channel,
            loop=loop,
            state=state,
            block=block_first,
        )

    threading.Thread(target=_bg, daemon=True).start()


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
        _attach_sink(ctx, channel, loop)
        graph = build_plan_graph(ctx)
        state = load_plan_state(workspace, thread_id) or {}
        try:
            new_state, intr = resume_after_confirm_decision(ctx, graph, state, decision)
            _emit_state(channel, loop, workspace, thread_id)
            if intr is not None:
                HUB.publish_sync(channel, {"type": "interrupt", "payload": intr}, loop)
            else:
                HUB.publish_sync(channel, {"type": "plan_done"}, loop)
        except Exception as exc:
            HUB.publish_sync(channel, {"type": "error", "message": str(exc)}, loop)
        finally:
            LOCKS.release(thread_id, owner="web")
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
            return
        ctx = _build_plan_ctx(workspace, thread_id, allow_write=allow_write)
        _attach_sink(ctx, channel, loop)
        graph = build_plan_graph(ctx)
        state = load_plan_state(workspace, thread_id) or {}
        try:
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
            snap = graph.get_state({"configurable": {"thread_id": thread_id}})
            if snap and snap.interrupts:
                new_state, intr = run_until_interrupt(
                    ctx,
                    graph,
                    {},
                    input_payload=Command(resume={}),
                )
                _emit_state(channel, loop, workspace, thread_id)
                if intr is not None:
                    HUB.publish_sync(channel, {"type": "interrupt", "payload": intr}, loop)
                else:
                    HUB.publish_sync(channel, {"type": "plan_done"}, loop)
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
            LOCKS.release(thread_id, owner="web")
            HUB.publish_sync(channel, {"type": "end"}, loop)

    if is_running(thread_id):
        HUB.publish_sync(channel, {"type": "error", "message": "Plan 已在运行"}, loop)
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
        _attach_sink(ctx, channel, loop)
        graph = build_plan_graph(ctx)
        state = load_plan_state(workspace, thread_id) or {}
        try:
            state = prepare_plan_for_resume(ctx, state)
            plan = state.get("plan") if isinstance(state.get("plan"), dict) else {}
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
            LOCKS.release(thread_id, owner="web")
            HUB.publish_sync(channel, {"type": "end"}, loop)

    if is_running(thread_id):
        HUB.publish_sync(channel, {"type": "error", "message": "Plan 已在运行"}, loop)
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
        state = load_plan_state(workspace, thread_id) or {}
        try:
            HUB.publish_sync(channel, {"type": "turn_start", "thread_id": thread_id})
            text = run_plan_discuss(ctx, state, message)
            HUB.publish_sync(
                channel,
                {"type": "turn_done", "text": text, "thread_id": thread_id},
                loop,
            )
        except Exception as exc:
            HUB.publish_sync(channel, {"type": "error", "message": str(exc)}, loop)
        finally:
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


def wait_plan_done(thread_id: str, timeout: float = 3600.0) -> bool:
    """
    等待 Plan 后台任务结束。

    @param thread_id plan thread
    @param timeout 超时秒
    @return 是否在超时前结束
    """
    return wait_until_done(thread_id, timeout=timeout)
