"""Agent 会话服务。"""

from __future__ import annotations

import asyncio
import threading
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
from llgraph.session.session_meta import save_session_meta
from llgraph.survey.edit_confirm import EditConfirmGate

from llgraph.console.runtime.event_hub import HUB
from llgraph.console.runtime.session_lock import LOCKS
from llgraph.console.runtime.sse_sink import SseTraceSink
from llgraph.console.runtime.workspace_runtime import RUNTIME_MANAGER


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
    channel: str = ""


def run_agent_chat(
    req: AgentChatRequest,
    *,
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[dict[str, Any]],
) -> None:
    """
    在后台线程执行 Agent 一轮对话。

    @param req 请求
    @param loop asyncio 循环
    @param queue 事件队列
    """

    channel = req.channel or f"agent:{req.thread_id}"

    def emit(event: dict[str, Any]) -> None:
        # 仅经 EventHub 投递；subscribe 的 queue 已在 HUB 中，勿重复 put_nowait
        HUB.publish_sync(channel, event, loop)

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

    rt = RUNTIME_MANAGER.get(req.workspace, allow_write=req.allow_write)
    trace = TraceSession(mode=rt.trace_session.mode)
    trace.trace_sink = SseTraceSink(emit)

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
        agent_ctx = build_agent_session_for_thread(bundle, req.thread_id)
        context_spill = ContextSpill.create(req.workspace, session_id=req.thread_id, disabled=False)
        edit_confirm_gate = EditConfirmGate(req.workspace) if req.allow_write else None

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
        )

        from llgraph.context.message_normalize import _message_text
        from llgraph.survey.survey_prompt import strip_survey_for_display

        raw_display = _message_text(text).strip() or (text.strip() if isinstance(text, str) else "")
        display_text = strip_survey_for_display(raw_display)
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
        sink = trace.trace_sink
        log_lines = list(getattr(sink, "log_lines", []) or [])
        if step_payloads or log_lines:
            from llgraph.session.web_trace_store import save_last_web_trace

            save_last_web_trace(
                req.workspace,
                req.thread_id,
                log_lines=log_lines,
                steps=step_payloads,
            )
        survey = _survey_payload(trace.last_turn_raw_reply or text)
        if survey is not None:
            payload["survey"] = survey
            payload["type"] = "survey"
        emit(payload)
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})
    finally:
        LOCKS.release(req.thread_id, owner="web")
        emit({"type": "end"})


def start_agent_chat_async(
    req: AgentChatRequest,
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[dict[str, Any]],
) -> threading.Thread:
    """
    异步启动 Agent 对话线程。

    @param req 请求
    @param loop 事件循环
    @param queue 队列
    @return 线程
    """
    thread = threading.Thread(
        target=run_agent_chat,
        args=(req,),
        kwargs={"loop": loop, "queue": queue},
        daemon=True,
    )
    thread.start()
    return thread
