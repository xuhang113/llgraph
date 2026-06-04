"""会话内只读/可写模式切换（/write on|off）。"""

from __future__ import annotations

from collections.abc import Callable

from llgraph.core.agent import rebuild_agent_preserving_memory
from llgraph.core.agent_session import AgentSessionContext
from llgraph.context.context_session import ContextSession
from llgraph.config.edit_settings import resolve_edit_settings
from llgraph.session.session_edits import SessionEditTracker
from llgraph.core.write_failure_tracker import WriteFailureTracker


def format_write_mode_status(agent_session: AgentSessionContext) -> str:
    """
    当前文件写入模式说明。

    @param agent_session Agent 会话
    @return 多行文本
    """
    if agent_session.allow_write:
        mode = "可写（search_replace / write_file / append_file 已启用）"
    else:
        mode = "只读（禁止写文件与受限 shell）"
    lines = [
        f"文件写入: {mode}",
        "命令: /write on  |  /write off",
    ]
    if agent_session.allow_write and agent_session.edit_tracker is not None:
        paths = agent_session.edit_tracker.unique_paths()
        if paths:
            lines.append(f"本会话已改 {len(paths)} 个文件（/changes · /undo）")
    return "\n".join(lines)


def set_session_write_mode(
    agent_session: AgentSessionContext,
    *,
    enabled: bool,
    context_session: ContextSession,
) -> bool:
    """
    切换会话写入模式并重建 Agent（保留对话历史）。

    @param agent_session Agent 会话
    @param enabled 是否可写
    @param context_session Rule/Skill 状态（写失败提醒）
    @return 是否发生切换（False 表示已是目标模式）
    """
    if agent_session.allow_write == enabled:
        return False

    workspace = agent_session.workspace
    edit_settings = resolve_edit_settings(workspace)

    if enabled:
        if agent_session.edit_tracker is None:
            agent_session.edit_tracker = SessionEditTracker(
                workspace,
                session_id=agent_session.thread_id,
            )
        if agent_session.write_failure_tracker is None:
            agent_session.write_failure_tracker = WriteFailureTracker(
                context_session,
                failures_before_hint=edit_settings.write_failures_before_hint,
                chunk_max_chars=edit_settings.write_chunk_max_chars,
            )

    mcp_tools = agent_session.mcp_tools
    registry = agent_session.mcp_registry
    if registry is not None:
        mcp_tools = registry.rebuild_for_allow_write(workspace, enabled)
        agent_session.mcp_tools = mcp_tools

    on_changed: Callable[[str], None] | None = None
    if enabled:
        on_changed = agent_session.on_file_changed

    rebuild_agent_preserving_memory(
        agent_session,
        allow_write=enabled,
        mcp_tools=mcp_tools,
        on_file_changed=on_changed,
    )
    agent_session.allow_write = enabled
    return True
