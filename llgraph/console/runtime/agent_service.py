"""Agent 会话服务。"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from llgraph.config.edit_settings import resolve_edit_settings
from llgraph.context.context_spill import ContextSpill
from llgraph.core.agent import invoke_agent
from llgraph.core.session_bootstrap import AgentRuntimeBundle, build_agent_session_for_thread
from llgraph.core.write_failure_tracker import WriteFailureTracker
from llgraph.display.trace_display import TraceSession
from llgraph.session.session_meta import save_session_meta, touch_session_activity

from llgraph.console.runtime.event_hub import HUB
from llgraph.console.runtime.session_lock import LOCKS
from llgraph.console.runtime.sse_sink import SseTraceSink
from llgraph.console.runtime.workspace_runtime import RUNTIME_MANAGER

_ACTIVE_AGENT_CHATS: set[str] = set()
_ACTIVE_AGENT_CHATS_LOCK = threading.Lock()
_CANCEL_REQUESTS: set[str] = set()
_CANCEL_LOCK = threading.Lock()
_TRACE_HEARTBEAT_SEC = 5.0


def _collect_web_trace_payload(trace: TraceSession) -> tuple[list[str], list[dict[str, Any]]]:
    """@return (log_lines, step_payloads)"""
    from llgraph.console.runtime.sse_sink import _step_to_dict

    sink = trace.trace_sink
    log_lines = list(getattr(sink, "log_lines", []) or [])
    step_payloads: list[dict[str, Any]] = []
    if trace.last_turn_steps:
        step_payloads = [_step_to_dict(s) for s in trace.last_turn_steps]
    elif sink is not None:
        inner = getattr(sink, "_inner", sink)
        payloads = getattr(sink, "_step_payloads", None)
        if isinstance(payloads, list) and payloads:
            step_payloads = list(payloads)
    return log_lines, step_payloads


def _persist_web_trace_turn(
    req: AgentChatRequest,
    trace: TraceSession,
    *,
    incomplete: bool = False,
    stop_reason: str | None = None,
    outcome: str | None = None,
) -> None:
    """
    将当前轮 trace 落盘（成功或中断均调用；终止原因以 execution/last_run 为准）。

    @param req 对话请求
    @param trace trace 会话
    @param incomplete 是否未完成
    @param stop_reason 终止原因摘要
    @param outcome ok | cancelled | error
    """
    log_lines, step_payloads = _collect_web_trace_payload(trace)
    if not step_payloads and not log_lines:
        return
    from llgraph.session.web_trace_store import save_last_web_trace

    save_last_web_trace(
        req.workspace,
        req.thread_id,
        log_lines=log_lines,
        steps=step_payloads,
        incomplete=incomplete,
        stop_reason=stop_reason,
        outcome=outcome,
    )
    touch_session_activity(req.workspace, req.thread_id)


def _trace_log_line_count(trace: TraceSession) -> int:
    sink = trace.trace_sink
    if sink is None:
        return 0
    inner = getattr(sink, "_inner", sink)
    lines = getattr(inner, "log_lines", None)
    return len(lines) if isinstance(lines, list) else 0


def _start_web_trace_heartbeat(
    trace: TraceSession,
    *,
    emit: Callable[[dict[str, Any]], None],
    stop: threading.Event,
    turn_start: float,
) -> threading.Thread:
    """
    Web 长耗时阶段心跳：工具/LLM 阻塞时仍推送 trace_activity，避免 UI 看似卡死。

    @param trace trace 会话
    @param emit SSE 回调
    @param stop 停止事件
    @param turn_start 轮次开始 monotonic 时间
    @return 心跳线程
    """

    def run() -> None:
        if trace.is_silent():
            return
        last_line_count = 0
        while not stop.wait(_TRACE_HEARTBEAT_SEC):
            lines_now = _trace_log_line_count(trace)
            if lines_now > last_line_count:
                last_line_count = lines_now
                continue
            elapsed = max(0, int(time.monotonic() - turn_start))
            if elapsed < int(_TRACE_HEARTBEAT_SEC):
                continue
            emit({"type": "trace_activity", "elapsed_sec": elapsed, "phase": "running"})

    thread = threading.Thread(target=run, daemon=True, name="web-trace-heartbeat")
    thread.start()
    return thread


def is_agent_chat_running(thread_id: str) -> bool:
    """@param thread_id 会话 ID @return Web Agent 后台线程是否在跑"""
    with _ACTIVE_AGENT_CHATS_LOCK:
        return thread_id in _ACTIVE_AGENT_CHATS


def try_register_agent_chat(thread_id: str) -> bool:
    """
    原子注册进行中的 Web Agent 对话（同 thread 不可重入）。

    @param thread_id 会话 ID
    @return 是否成功注册
    """
    with _ACTIVE_AGENT_CHATS_LOCK:
        if thread_id in _ACTIVE_AGENT_CHATS:
            return False
        _ACTIVE_AGENT_CHATS.add(thread_id)
        return True


def request_agent_cancel(thread_id: str) -> bool:
    """
    请求停止进行中的 Web Agent 对话（在 ReAct 步间生效）。

    @param thread_id 会话 ID
    @return 是否已标记（False 表示当前无后台对话）
    """
    with _ACTIVE_AGENT_CHATS_LOCK:
        running = thread_id in _ACTIVE_AGENT_CHATS
    if not running:
        return False
    with _CANCEL_LOCK:
        _CANCEL_REQUESTS.add(thread_id)
    return True


def is_agent_cancel_requested(thread_id: str) -> bool:
    """@param thread_id 会话 ID @return 是否已请求停止"""
    with _CANCEL_LOCK:
        return thread_id in _CANCEL_REQUESTS


def clear_agent_cancel(thread_id: str) -> None:
    """@param thread_id 会话 ID"""
    with _CANCEL_LOCK:
        _CANCEL_REQUESTS.discard(thread_id)


def abort_agent_chat(thread_id: str) -> dict[str, Any]:
    """
    Web Stop：标记取消，ReAct stream 在下一步前退出。

    @param thread_id 会话 ID
    @return ok / message
    """
    if not request_agent_cancel(thread_id):
        return {"ok": False, "message": "当前无进行中的 Agent 对话"}
    return {"ok": True, "message": "已请求停止"}


def create_agent_session(workspace: Path, *, title: str = "") -> str:
    """
    创建新 Agent 会话。

    @param workspace 工作区根
    @param title 可选标题
    @return cli-* thread_id
    """
    thread_id = f"cli-{uuid.uuid4().hex[:8]}"
    save_session_meta(
        workspace,
        thread_id,
        {
            "session_kind": "agent",
            "workspace": str(workspace.expanduser().resolve()),
            "title": title or "",
        },
    )
    return thread_id


def _survey_spec_to_dict(spec: Any) -> dict[str, Any]:
    questions = []
    for q in spec.questions:
        questions.append(
            {
                "id": q.question_id,
                "prompt": q.prompt,
                "options": list(q.options),
                "default_index": q.default_index,
                "default_indices": list(q.default_indices),
                "allow_free_text": q.allow_free_text,
                "step_label": q.step_label,
                "option_hints": list(q.option_hints),
                "multi_select": q.multi_select,
            }
        )
    return {"title": spec.title, "questions": questions}


def _survey_payload(text: str) -> dict[str, Any] | None:
    from llgraph.survey.survey_prompt import resolve_survey_from_assistant

    spec = resolve_survey_from_assistant(text)
    if spec is None:
        return None
    return _survey_spec_to_dict(spec)


@dataclass
class AgentChatRequest:
    """Agent 对话请求。"""

    workspace: Path
    thread_id: str
    message: str
    allow_write: bool = False
    images: list | None = None


def run_agent_chat(
    req: AgentChatRequest,
    *,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """
    在后台线程执行 Agent 一轮对话。

    @param req 请求
    @param loop asyncio 循环
    """

    def emit(event: dict[str, Any]) -> None:
        payload = {**event, "thread_id": req.thread_id}
        HUB.publish_sync(f"session:{req.thread_id}", payload, loop)

    if is_agent_chat_running(req.thread_id):
        emit(
            {
                "type": "error",
                "message": "Agent 对话进行中，请等待结束或先停止",
            }
        )
        emit({"type": "end"})
        return

    ok, holder = LOCKS.acquire(req.thread_id, owner="web")
    if not ok and holder is not None:
        emit(
            {
                "type": "error",
                "message": f"会话 {req.thread_id} 正被 {holder.owner} 占用，请勿与 CLI 同时操作",
            }
        )
        emit({"type": "end"})
        return

    if not try_register_agent_chat(req.thread_id):
        LOCKS.release(req.thread_id, owner="web")
        emit(
            {
                "type": "error",
                "message": "Agent 对话进行中，请等待结束或先停止",
            }
        )
        emit({"type": "end"})
        return

    rt = RUNTIME_MANAGER.get(req.workspace, allow_write=req.allow_write)
    trace = TraceSession(mode=rt.trace_session.mode)
    from llgraph.console.runtime.sse_sink import PersistingSseTraceSink, SseTraceSink

    inner = SseTraceSink(emit)
    trace.trace_sink = PersistingSseTraceSink(
        inner,
        workspace=req.workspace,
        thread_id=req.thread_id,
    )

    edit_settings = resolve_edit_settings(req.workspace)
    write_failure_tracker = (
        WriteFailureTracker(
            rt.context_session,
            failures_before_hint=edit_settings.write_failures_before_hint,
            chunk_max_chars=edit_settings.write_chunk_max_chars,
        )
        if req.allow_write
        else None
    )
    bundle = AgentRuntimeBundle(
        workspace=req.workspace,
        trace_session=trace,
        context_session=rt.context_session,
        allow_write=req.allow_write,
        mcp_tools=rt.mcp_tools,
        mcp_registry=rt.mcp_registry,
        watch_service=None,
        web_search_enabled=rt.web_search_enabled,
        sandbox_policy=rt.sandbox_policy,
        sandbox_cli_enabled=rt.sandbox_cli_enabled,
        no_spill=False,
        memory_kind="memory",
        mcp_summary=rt.mcp_summary,
        watch_active=False,
    )

    try:
        emit({"type": "turn_start", "thread_id": req.thread_id})
        touch_session_activity(req.workspace, req.thread_id)
        agent_ctx = build_agent_session_for_thread(
            bundle,
            req.thread_id,
        )
        context_spill = ContextSpill.create(req.workspace, session_id=req.thread_id, disabled=False)

        hb_stop = threading.Event()
        turn_start = time.monotonic()
        _start_web_trace_heartbeat(
            trace,
            emit=emit,
            stop=hb_stop,
            turn_start=turn_start,
        )
        try:
            def cancel_check() -> bool:
                return is_agent_cancel_requested(req.thread_id)

            text = invoke_agent(
                agent_ctx.agent,
                req.message,
                workspace_root=req.workspace,
                thread_id=req.thread_id,
                with_memory=True,
                trace_session=trace,
                context_session=rt.context_session,
                write_failure_tracker=write_failure_tracker,
                context_spill=context_spill,
                allow_write=req.allow_write,
                cancel_check=cancel_check,
                run_source="web",
                images=req.images,
            )
        finally:
            hb_stop.set()

        cancelled = is_agent_cancel_requested(req.thread_id)
        if cancelled:
            clear_agent_cancel(req.thread_id)
            _persist_web_trace_turn(
                req,
                trace,
                incomplete=True,
                stop_reason="用户停止当前生成",
                outcome="cancelled",
            )
            emit(
                {
                    "type": "interrupt",
                    "payload": {
                        "type": "user_stop",
                        "message": "用户已停止当前生成。",
                    },
                }
            )
        else:
            from llgraph.context.message_normalize import _message_text, format_agent_chat_display_text
            from llgraph.survey.survey_prompt import strip_survey_for_display

            raw_display = _message_text(text).strip() or (text.strip() if isinstance(text, str) else "")
            display_text = format_agent_chat_display_text(strip_survey_for_display(raw_display))
            payload: dict[str, Any] = {
                "type": "turn_done",
                "text": display_text,
                "thread_id": req.thread_id,
            }
            step_payloads: list[dict[str, Any]] = []
            if trace.last_turn_steps:
                from llgraph.console.runtime.sse_sink import _step_to_dict

                step_payloads = [_step_to_dict(s) for s in trace.last_turn_steps]
                payload["trace_steps"] = step_payloads
            log_lines, _ = _collect_web_trace_payload(trace)
            if step_payloads or log_lines:
                from llgraph.session.web_trace_store import save_last_web_trace

                save_last_web_trace(
                    req.workspace,
                    req.thread_id,
                    log_lines=log_lines,
                    steps=step_payloads,
                    outcome="ok",
                )
                touch_session_activity(req.workspace, req.thread_id)
            survey = None
            if req.allow_write:
                from llgraph.config.survey_settings import survey_followup_enabled

                if survey_followup_enabled(req.workspace, rt.context_session):
                    survey = _survey_payload(trace.last_turn_raw_reply or text)
            if survey is not None:
                payload["survey"] = survey
                payload["type"] = "survey"
            emit(payload)
    except Exception as exc:
        _persist_web_trace_turn(
            req,
            trace,
            incomplete=True,
            stop_reason=str(exc).strip() or type(exc).__name__,
            outcome="error",
        )
        emit({"type": "error", "message": str(exc)})
    finally:
        clear_agent_cancel(req.thread_id)
        with _ACTIVE_AGENT_CHATS_LOCK:
            _ACTIVE_AGENT_CHATS.discard(req.thread_id)
        LOCKS.release(req.thread_id, owner="web")
        emit({"type": "end"})


def start_agent_chat_async(
    req: AgentChatRequest,
    loop: asyncio.AbstractEventLoop,
) -> threading.Thread:
    """
    异步启动 Agent 对话线程。

    @param req 请求
    @param loop 事件循环
    @return 线程
    """
    thread = threading.Thread(
        target=run_agent_chat,
        args=(req,),
        kwargs={"loop": loop},
        daemon=True,
    )
    thread.start()
    return thread
