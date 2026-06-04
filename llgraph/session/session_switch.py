"""会话内切换 thread_id（/session use|new）。"""

from __future__ import annotations

import shlex
import uuid
from pathlib import Path

from llgraph.core.agent_session import AgentSessionContext
from llgraph.context.context_spill import ContextSpill
from llgraph.config.edit_settings import resolve_edit_settings
from llgraph.session.session_edits import SessionEditTracker
from llgraph.session.session_manifest import sync_session_manifest_to_agent_state
from llgraph.session.session_file_store import restore_session_to_agent
from llgraph.session.session_registry import format_sessions_list, session_is_resumable
from llgraph.core.write_failure_tracker import WriteFailureTracker


def switch_agent_thread(
    agent_session: AgentSessionContext,
    new_thread_id: str,
    *,
    no_spill: bool = False,
) -> tuple[bool, str]:
    """
    切换当前交互会话 thread_id，并重建 spill / 编辑账本 / manifest。

    @param agent_session Agent 会话
    @param new_thread_id 目标 thread_id
    @param no_spill 是否禁用工具落盘
    @return (是否发生切换, 提示信息)
    """
    new_thread_id = new_thread_id.strip()
    if not new_thread_id:
        return False, "thread_id 不能为空。"

    if agent_session.thread_id == new_thread_id:
        return False, f"当前已是会话 {new_thread_id}。"

    workspace = agent_session.workspace
    allow_write = agent_session.allow_write

    agent_session.thread_id = new_thread_id
    agent_session.context_spill = ContextSpill.create(
        workspace,
        session_id=new_thread_id,
        disabled=no_spill,
    )
    agent_session.edit_tracker = (
        SessionEditTracker(workspace, session_id=new_thread_id) if allow_write else None
    )

    if allow_write and agent_session.write_failure_tracker is not None:
        edit_settings = resolve_edit_settings(workspace)
        agent_session.write_failure_tracker = WriteFailureTracker(
            agent_session.context_session,
            failures_before_hint=edit_settings.write_failures_before_hint,
            chunk_max_chars=edit_settings.write_chunk_max_chars,
        )

    _ok, resume_hint = session_is_resumable(workspace, new_thread_id)
    sync_session_manifest_to_agent_state(
        agent_session.agent,
        thread_id=new_thread_id,
        workspace=workspace,
        session=agent_session.context_session,
        user_message="",
        with_memory=agent_session.with_memory,
    )

    msg_count = 0
    if agent_session.with_memory:
        msg_count = restore_session_to_agent(
            agent_session.agent, workspace, new_thread_id
        )

    parts = [f"已切换到会话 {new_thread_id}。", resume_hint]
    if msg_count > 1:
        parts.append(f"已加载 {msg_count} 条历史消息。")
    return True, " ".join(parts)


def create_new_thread_id() -> str:
    """
    生成新会话 ID。

    @return cli-xxxxxxxx 形式
    """
    return f"cli-{uuid.uuid4().hex[:8]}"


def format_resume_cli_command(workspace: Path, thread_id: str) -> str:
    """
    生成从 shell 恢复当前会话的 llgraph 命令。

    @param workspace 工作区根
    @param thread_id 会话 thread_id
    @return 可复制的命令行
    """
    ws = str(workspace.expanduser().resolve())
    return f"llgraph -C {shlex.quote(ws)} --thread-id {thread_id}"


def print_current_session_info(workspace: Path, thread_id: str) -> None:
    """
    打印当前会话 ID 与恢复方式（/session current、/sessionid）。

    @param workspace 工作区根
    @param thread_id 当前 thread_id
    """
    from llgraph.session.session_meta import resolve_session_display_title
    from llgraph.session.user_storage import format_storage_location_hint

    title = resolve_session_display_title(workspace, thread_id)
    print(f"当前会话 thread_id: {thread_id}", flush=True)
    if title and title != thread_id:
        print(f"标题: {title}", flush=True)
    print(format_storage_location_hint(workspace), flush=True)
    print(f"恢复: {format_resume_cli_command(workspace, thread_id)}", flush=True)
    print(f"会话内切换: /session use {thread_id}", flush=True)


def print_session_exit_hint(workspace: Path, thread_id: str) -> None:
    """
    交互退出时提示 session id 与恢复命令。

    @param workspace 工作区根
    @param thread_id 当前 thread_id
    """
    from llgraph.session.session_meta import resolve_session_display_title

    title = resolve_session_display_title(workspace, thread_id)
    title_part = f"（{title}）" if title and title != thread_id else ""
    print("", flush=True)
    print(f"当前会话 thread_id: {thread_id} {title_part}".rstrip(), flush=True)
    print(f"下次恢复: {format_resume_cli_command(workspace, thread_id)}", flush=True)


def format_session_command_help(workspace: Path, current_thread_id: str) -> str:
    """
    /session 子命令说明。

    @param workspace 工作区根
    @param current_thread_id 当前 thread_id
    @return 多行文本
    """
    lines = [
        "会话切换（thread_id）",
        "",
        format_sessions_list(workspace, current_thread_id=current_thread_id),
        "",
        "命令:",
        "  /session              列出会话（同 /sessions）",
        "  /session use <id>     切换到已有会话",
        "  /session new          新建并切换到新会话",
        "  /session title <标题>  重命名当前会话（手动，不会被自动覆盖）",
        "  /session title <id> <标题>  重命名指定会话",
        "  /session delete <id>  删除指定会话",
        "  /session delete all   删除除当前外全部",
        "  /session delete all --including-current  全量删除",
        "  /session current      显示当前 thread_id 与恢复命令",
        "  /session id           同 /session current",
        "  /sessionid            同 /session current",
    ]
    return "\n".join(lines)
