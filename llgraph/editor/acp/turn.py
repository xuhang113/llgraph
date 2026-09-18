"""ACP 一轮对话：复用 Console 的工作区 Runtime 与会话保活池。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

def request_turn_cancel(thread_id: str) -> bool:
    """
    请求停止进行中的一轮。

    进行中的 LLM 调用只认 Console 的取消登记表（``react_invoke`` 从那里读），
    光把 ACP 会话上的 cancelled 置位只能在步与步之间生效。

    @param thread_id 会话 ID
    @return 是否标记成功（当前无进行中对话时为 False）
    """
    from llgraph.console.runtime.agent_service import request_agent_cancel

    return request_agent_cancel(thread_id)


@dataclass
class AcpTurnRequest:
    """一轮 ACP 提问。"""

    workspace: Path
    thread_id: str
    message: str
    allow_write: bool = False


@dataclass
class AcpTurnResult:
    """一轮执行结果。"""

    text: str = ""
    stop_reason: str = "end_turn"


def run_acp_turn(
    req: AcpTurnRequest,
    *,
    send_update: Callable[[dict[str, Any]], None],
    cancel_check: Callable[[], bool],
) -> AcpTurnResult:
    """
    执行一轮 ReAct，并把过程实时推成 ``session/update``。

    与 Web Console 走同一套 Runtime / 会话保活池 / messages.jsonl，
    因此同一个 thread 在编辑器里聊到一半可以切回 CLI 继续。

    @param req 请求
    @param send_update ``session/update`` 的 update 字段回调
    @param cancel_check 返回 True 时中断本轮
    @return 结果（stop_reason 为 end_turn 或 cancelled）
    """
    from llgraph.config.edit_settings import resolve_edit_settings
    from llgraph.console.runtime.agent_service import (
        force_release_agent_chat,
        is_agent_cancel_requested,
        try_register_agent_chat,
    )
    from llgraph.console.runtime.session_lock import LOCKS
    from llgraph.console.runtime.workspace_runtime import RUNTIME_MANAGER
    from llgraph.context.context_spill import ContextSpill
    from llgraph.core.agent import invoke_agent
    from llgraph.core.session_bootstrap import (
        AgentRuntimeBundle,
        get_or_build_agent_session_for_thread,
    )
    from llgraph.core.write_failure_tracker import WriteFailureTracker
    from llgraph.display.trace_display import TraceSession
    from llgraph.editor.acp.sink import AcpTraceSink
    from llgraph.editor.acp.updates import agent_message_chunk
    from llgraph.session.session_meta import touch_session_activity

    ok, holder = LOCKS.acquire(req.thread_id, owner="acp")
    if not ok and holder is not None:
        raise RuntimeError(
            f"会话 {req.thread_id} 正被 {holder.owner} 占用，请勿与 CLI / Web 同时操作"
        )
    if not try_register_agent_chat(req.thread_id):
        LOCKS.release(req.thread_id, owner="acp")
        raise RuntimeError("该会话已有对话在进行")

    rt = RUNTIME_MANAGER.get(req.workspace, allow_write=req.allow_write)
    # MCP 后台加载；短等即可，未就绪则本轮无 MCP 工具（与 Web 一致）
    RUNTIME_MANAGER.wait_mcp_ready(req.workspace, timeout=2.0)

    trace = TraceSession(mode=rt.trace_session.mode)
    sink = AcpTraceSink(send_update)
    trace.trace_sink = sink

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

    cancelled = False
    try:
        touch_session_activity(req.workspace, req.thread_id)
        agent_ctx = get_or_build_agent_session_for_thread(bundle, req.thread_id)
        context_spill = ContextSpill.create(
            req.workspace,
            session_id=req.thread_id,
            disabled=False,
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
            run_source="acp",
        )
        cancelled = cancel_check() or is_agent_cancel_requested(req.thread_id)
    finally:
        # 注册表与 session 锁一起释放：漏一个，这个 thread 之后就再也开不了新一轮
        if not force_release_agent_chat(req.thread_id, owner="acp"):
            LOCKS.release(req.thread_id, owner="acp")

    if cancelled:
        return AcpTurnResult(text="", stop_reason="cancelled")

    from llgraph.context.message_normalize import (
        _message_text,
        format_agent_chat_display_text,
    )

    raw = _message_text(text).strip() or (text.strip() if isinstance(text, str) else "")
    display_text = format_agent_chat_display_text(raw)
    # 模型没走流式（或本轮被 trace 模式静默）时，正文只剩这一处可发
    if display_text and sink.streamed_chars <= 0:
        send_update(agent_message_chunk(display_text))
    return AcpTurnResult(text=display_text, stop_reason="end_turn")
