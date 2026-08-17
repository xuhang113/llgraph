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
from llgraph.core.session_bootstrap import AgentRuntimeBundle, get_or_build_agent_session_for_thread
from llgraph.core.write_failure_tracker import WriteFailureTracker
from llgraph.display.trace_display import TraceSession
from llgraph.session.session_meta import save_session_meta, touch_session_activity

from llgraph.console.runtime.event_hub import HUB
from llgraph.console.runtime.session_lock import LOCKS
from llgraph.console.runtime.sse_sink import SseTraceSink
from llgraph.console.runtime.workspace_runtime import RUNTIME_MANAGER

_ACTIVE_AGENT_CHATS: set[str] = set()
_ACTIVE_AGENT_CHAT_THREADS: dict[str, threading.Thread] = {}
_ACTIVE_AGENT_CHATS_LOCK = threading.Lock()
_STALE_TRACE_IDLE_SEC = 180.0
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
    from llgraph.session.web_trace_store import append_web_trace_turn

    append_web_trace_turn(
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
    workspace: Path | None = None,
    thread_id: str | None = None,
) -> threading.Thread:
    """
    Web 长耗时阶段心跳：工具/LLM 阻塞时仍推送 trace_activity，避免 UI 看似卡死。

    @param trace trace 会话
    @param emit SSE 回调
    @param stop 停止事件
    @param turn_start 轮次开始 monotonic 时间
    @param workspace 工作区（用于刷新 live_web_trace mtime）
    @param thread_id 会话 thread
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
            phase = getattr(trace, "react_phase", "") or "running"
            emit(
                {
                    "type": "trace_activity",
                    "elapsed_sec": elapsed,
                    "phase": phase,
                }
            )
            if workspace is not None and thread_id:
                try:
                    from llgraph.session.web_trace_store import touch_live_web_trace

                    touch_live_web_trace(workspace, thread_id)
                except Exception:
                    pass

    thread = threading.Thread(target=run, daemon=True, name="web-trace-heartbeat")
    thread.start()
    return thread


def _cleanup_agent_chat_registration(thread_id: str) -> None:
    """@param thread_id 会话 ID"""
    with _ACTIVE_AGENT_CHATS_LOCK:
        _ACTIVE_AGENT_CHATS.discard(thread_id)
        _ACTIVE_AGENT_CHAT_THREADS.pop(thread_id, None)


def force_release_agent_chat(thread_id: str, *, owner: str = "web") -> bool:
    """
    强制释放僵死 Web Agent 对话占用（锁 + 注册表），便于用户恢复继续提问。

    @param thread_id 会话 ID
    @param owner 锁持有者
    @return 是否曾处于 running
    """
    with _ACTIVE_AGENT_CHATS_LOCK:
        was_running = thread_id in _ACTIVE_AGENT_CHATS
    if was_running:
        clear_agent_cancel(thread_id)
        _cleanup_agent_chat_registration(thread_id)
        LOCKS.release(thread_id, owner=owner)
    return was_running


def reconcile_stale_agent_chat(
    workspace: Path,
    thread_id: str,
    *,
    max_trace_idle_sec: float = _STALE_TRACE_IDLE_SEC,
) -> bool:
    """
    若 Web 会话登记为 running、但工作线程已死且 live_web_trace 长时间无更新，则强制释放占用。

    注意：spawn_subagent 等长工具阻塞时父会话可能数分钟无新步骤，只要工作线程仍存活就不得误杀，
    否则 sessionMeta.running=false，前端切回标签会以为 ReAct 已终止。

    @param workspace 工作区根
    @param thread_id 会话 ID
    @param max_trace_idle_sec live_web_trace 空闲秒数阈值
    @return 是否执行了强制释放
    """
    if not is_agent_chat_running(thread_id):
        return False
    with _ACTIVE_AGENT_CHATS_LOCK:
        worker = _ACTIVE_AGENT_CHAT_THREADS.get(thread_id)
    if worker is not None and worker.is_alive():
        return False
    from llgraph.session.web_trace_store import live_web_trace_path

    path = live_web_trace_path(workspace, thread_id)
    if not path.is_file():
        return False
    try:
        idle = time.time() - path.stat().st_mtime
    except OSError:
        return False
    if idle < max_trace_idle_sec:
        return False
    return force_release_agent_chat(thread_id)


def is_agent_chat_running(thread_id: str) -> bool:
    """@param thread_id 会话 ID @return Web Agent 后台线程是否在跑"""
    with _ACTIVE_AGENT_CHATS_LOCK:
        if thread_id not in _ACTIVE_AGENT_CHATS:
            return False
        worker = _ACTIVE_AGENT_CHAT_THREADS.get(thread_id)
        if worker is not None and not worker.is_alive():
            _ACTIVE_AGENT_CHATS.discard(thread_id)
            _ACTIVE_AGENT_CHAT_THREADS.pop(thread_id, None)
            return False
        return True


def try_register_agent_chat(thread_id: str) -> bool:
    """
    原子注册进行中的 Web Agent 对话（同 thread 不可重入）。

    @param thread_id 会话 ID
    @return 是否成功注册
    """
    with _ACTIVE_AGENT_CHATS_LOCK:
        if thread_id in _ACTIVE_AGENT_CHATS:
            worker = _ACTIVE_AGENT_CHAT_THREADS.get(thread_id)
            if worker is not None and not worker.is_alive():
                _ACTIVE_AGENT_CHATS.discard(thread_id)
                _ACTIVE_AGENT_CHAT_THREADS.pop(thread_id, None)
            else:
                return False
        _ACTIVE_AGENT_CHATS.add(thread_id)
        return True


def request_agent_cancel(thread_id: str) -> bool:
    """
    请求停止进行中的 Web Agent 对话（立即中断当前 LLM invoke / 步间退出 ReAct）。

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
    Web Stop：标记取消并关闭进行中的 LLM stream，不等待 invoke 整包返回。

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
            "allow_write": True,
        },
        touch_activity=True,
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
    """本轮已落盘的图片引用；有则不再二次 save，且可跳过 pending append。"""
    image_refs: list | None = None
    skip_pending_user_append: bool = False


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
    # MCP 后台加载；短等即可。失败/超时不阻塞对话，未就绪则本轮无 MCP 工具。
    RUNTIME_MANAGER.wait_mcp_ready(req.workspace, timeout=2.0)
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
        watch_service=rt.watch_service,
        web_search_enabled=rt.web_search_enabled,
        sandbox_policy=rt.sandbox_policy,
        sandbox_cli_enabled=rt.sandbox_cli_enabled,
        no_spill=False,
        memory_kind="memory",
        mcp_summary=rt.mcp_summary,
        watch_active=bool(
            rt.watch_service is not None and getattr(rt.watch_service, "active", False)
        ),
    )

    turn_end_emitted = False
    turn_start = time.monotonic()
    agent_ctx = None

    try:
        emit({"type": "turn_start", "thread_id": req.thread_id})
        touch_session_activity(req.workspace, req.thread_id)
        agent_ctx = get_or_build_agent_session_for_thread(
            bundle,
            req.thread_id,
        )
        context_spill = ContextSpill.create(req.workspace, session_id=req.thread_id, disabled=False)

        hb_stop = threading.Event()
        _start_web_trace_heartbeat(
            trace,
            emit=emit,
            stop=hb_stop,
            turn_start=turn_start,
            workspace=req.workspace,
            thread_id=req.thread_id,
        )
        try:
            def cancel_check() -> bool:
                return is_agent_cancel_requested(req.thread_id)

            slot = getattr(agent_ctx, "subagent_parent_slot", None)
            if slot is not None:
                agent_ctx.sse_emit = emit
                slot.bind_from_session(
                    agent_ctx,
                    sse_emit=emit,
                    cancel_check=cancel_check,
                )

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
                image_refs=req.image_refs,
                skip_pending_user_append=req.skip_pending_user_append,
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
                "duration_sec": round(time.monotonic() - turn_start, 3),
            }
            step_payloads: list[dict[str, Any]] = []
            if trace.last_turn_steps:
                from llgraph.console.runtime.sse_sink import _step_to_dict

                step_payloads = [_step_to_dict(s) for s in trace.last_turn_steps]
                payload["trace_steps"] = step_payloads
            log_lines, _ = _collect_web_trace_payload(trace)
            if step_payloads or log_lines:
                from llgraph.session.web_trace_store import append_web_trace_turn

                append_web_trace_turn(
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
            from llgraph.session.session_title_llm import schedule_session_title_llm_refresh

            schedule_session_title_llm_refresh(
                req.workspace,
                req.thread_id,
                user_message=req.message,
                assistant_reply=display_text,
                on_updated=lambda title: emit({
                    "type": "title_updated",
                    "title": title,
                    "thread_id": req.thread_id,
                }),
            )
    except Exception as exc:
        force_release_agent_chat(req.thread_id)
        emit({"type": "error", "message": str(exc)})
        emit({"type": "end"})
        turn_end_emitted = True
        # 准备阶段/未进入 stream 的异常不会走 invoke 内 log_turn_failure；写入执行日志便于 Log 面板排查
        if not getattr(exc, "_llgraph_turn_error_logged", False):
            try:
                from llgraph.display.execution_log import log_turn_failure
                from llgraph.session.session_run_log import trace_run_context

                agent_for_log = None
                if agent_ctx is not None:
                    agent_for_log = getattr(agent_ctx, "agent", None)
                log_turn_failure(
                    req.workspace,
                    thread_id=req.thread_id,
                    with_memory=True,
                    agent=agent_for_log or object(),
                    duration_sec=time.monotonic() - turn_start,
                    error=exc,
                    outcome="error",
                    trace_context=trace_run_context(trace),
                    user_message=req.message,
                )
            except Exception:
                pass
        try:
            _persist_web_trace_turn(
                req,
                trace,
                incomplete=True,
                stop_reason=str(exc).strip() or type(exc).__name__,
                outcome="error",
            )
        except Exception:
            pass
    finally:
        clear_agent_cancel(req.thread_id)
        _cleanup_agent_chat_registration(req.thread_id)
        LOCKS.release(req.thread_id, owner="web")
        if not turn_end_emitted:
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
        name=f"web-agent-{req.thread_id[:12]}",
    )
    with _ACTIVE_AGENT_CHATS_LOCK:
        _ACTIVE_AGENT_CHAT_THREADS[req.thread_id] = thread
    thread.start()
    return thread
