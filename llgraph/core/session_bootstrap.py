"""交互会话 Agent 上下文构建（Agent / Plan 切换复用）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llgraph.config.edit_settings import resolve_edit_settings
from llgraph.context.context_session import ContextSession
from llgraph.context.context_spill import ContextSpill
from llgraph.core.agent import build_agent
from llgraph.core.agent_session import AgentSessionContext
from llgraph.core.write_failure_tracker import WriteFailureTracker
from llgraph.display.trace_display import TraceSession
from llgraph.session.session_edits import SessionEditTracker
from llgraph.session.session_file_store import restore_session_to_agent
from llgraph.survey.edit_confirm import EditConfirmGate


@dataclass
class AgentRuntimeBundle:
    """Agent 运行时依赖包。"""

    workspace: Path
    trace_session: TraceSession
    context_session: ContextSession
    allow_write: bool
    mcp_tools: list
    mcp_registry: Any
    watch_service: Any
    web_search_enabled: bool
    sandbox_policy: Any
    sandbox_cli_enabled: bool | None
    no_spill: bool
    memory_kind: str
    mcp_summary: str
    watch_active: bool


def build_agent_session_for_thread(
    bundle: AgentRuntimeBundle,
    thread_id: str,
) -> AgentSessionContext:
    """
    为指定 cli thread 构建 AgentSessionContext。

    @param bundle 运行时依赖
    @param thread_id cli-* thread
    @return AgentSessionContext
    """
    workspace = bundle.workspace
    allow_write = bundle.allow_write
    context_spill = ContextSpill.create(
        workspace,
        session_id=thread_id,
        disabled=bundle.no_spill,
    )
    edit_tracker = SessionEditTracker(workspace, session_id=thread_id)
    edit_settings = resolve_edit_settings(workspace)
    write_failure_tracker = (
        WriteFailureTracker(
            bundle.context_session,
            failures_before_hint=edit_settings.write_failures_before_hint,
            chunk_max_chars=edit_settings.write_chunk_max_chars,
        )
        if allow_write
        else None
    )
    edit_confirm_gate = EditConfirmGate(workspace) if allow_write else None
    watch_service = bundle.watch_service

    def on_file_changed(rel: str) -> None:
        if watch_service is not None and watch_service.active:
            watch_service.notify_changed(rel)

    agent = build_agent(
        with_memory=True,
        workspace_root=workspace,
        allow_write=allow_write,
        edit_tracker=edit_tracker if allow_write else None,
        on_file_changed=on_file_changed if allow_write else None,
        mcp_tools=bundle.mcp_tools,
        context_spill=context_spill,
        write_failure_tracker=write_failure_tracker,
        web_search_enabled=bundle.web_search_enabled,
        edit_confirm_gate=edit_confirm_gate,
        context_session=bundle.context_session,
        sandbox_policy=bundle.sandbox_policy,
    )
    restore_session_to_agent(agent, workspace, thread_id)
    return AgentSessionContext(
        agent=agent,
        workspace=workspace,
        thread_id=thread_id,
        trace_session=bundle.trace_session,
        context_session=bundle.context_session,
        with_memory=True,
        edit_tracker=edit_tracker,
        context_spill=context_spill,
        write_failure_tracker=write_failure_tracker,
        allow_write=allow_write,
        mcp_tools=bundle.mcp_tools,
        mcp_registry=bundle.mcp_registry,
        on_file_changed=on_file_changed if allow_write else None,
        watch_service=watch_service,
        web_search_enabled=bundle.web_search_enabled,
        edit_confirm_gate=edit_confirm_gate,
        sandbox_policy=bundle.sandbox_policy,
        sandbox_cli_enabled=bundle.sandbox_cli_enabled,
    )
