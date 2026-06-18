"""会话内只读/可写模式切换（/write on|off）。"""

from __future__ import annotations

from collections.abc import Callable

from llgraph.core.agent import rebuild_agent_preserving_memory
from llgraph.core.agent_session import AgentSessionContext
from llgraph.context.context_session import ContextSession
from llgraph.config.edit_settings import resolve_edit_settings
from llgraph.config.sandbox_settings import resolve_sandbox_settings
from llgraph.session.session_edits import SessionEditTracker
from llgraph.core.write_failure_tracker import WriteFailureTracker
from llgraph.sandbox.policy import build_sandbox_policy


def format_file_access_mode_label(allow_write: bool) -> str:
    """文件访问模式短标签。"""
    return "可读写" if allow_write else "只读"


def format_file_access_manifest_line(allow_write: bool) -> str:
    """
    置顶 <session-manifest> 中的系统属性行。

    @param allow_write 是否允许 Agent 写文件
    @return 单行说明
    """
    if allow_write:
        return (
            "系统属性 · 文件访问: 可读写"
            "（write_file / append_file / search_replace 可用；可落盘 tmp 模式 .tmp.md）"
        )
    return (
        "系统属性 · 文件访问: 只读"
        "（禁止 write_file、append_file、search_replace；无法落盘 tmp 模式或覆盖业务总览；"
        "梳理结果写在助手正文，或提示用户 /write on 后再落盘）"
    )


def format_file_access_workspace_context(allow_write: bool) -> str:
    """
    每轮 <workspace-context> 中的文件访问提示。

    @param allow_write 是否允许 Agent 写文件
    @return Markdown 块
    """
    label = format_file_access_mode_label(allow_write)
    if allow_write:
        return f"## 系统属性\n\n- 文件访问: **{label}**（可落盘 docs / tmp 模式 .tmp.md）"
    return (
        f"## 系统属性\n\n"
        f"- 文件访问: **{label}**（禁止写文件；用户确认中的 tmp/覆盖选项**不可执行**；"
        f"须在正文输出梳理或说明请先 `/write on`）"
    )


def format_write_mode_status(agent_session: AgentSessionContext) -> str:
    """
    当前文件写入模式说明。

    @param agent_session Agent 会话
    @return 多行文本
    """
    if agent_session.allow_write:
        mode = "可写（search_replace / write_file / append_file 已启用）"
    else:
        mode = "只读（禁止 Agent 写文件；/changes · /undo 仍可用）"
    lines = [
        f"文件写入: {mode}",
        "命令: /write on  |  /write off",
    ]
    policy = agent_session.sandbox_policy
    if policy is not None and policy.enabled:
        lines.append(
            f"OS 沙箱: {policy.mode}（{policy.backend}；随 /write 与 -w 联动）"
        )
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

    sandbox_settings = resolve_sandbox_settings(workspace)
    cli_sandbox = getattr(agent_session, "sandbox_cli_enabled", None)
    sandbox_policy = build_sandbox_policy(
        workspace,
        sandbox_settings,
        cli_enabled=cli_sandbox,
        allow_write=enabled,
    )
    agent_session.sandbox_policy = sandbox_policy

    rebuild_agent_preserving_memory(
        agent_session,
        allow_write=enabled,
        mcp_tools=mcp_tools,
        on_file_changed=on_changed,
        sandbox_policy=sandbox_policy,
    )
    agent_session.allow_write = enabled

    from llgraph.session.session_manifest import sync_session_manifest_to_agent_state

    sync_session_manifest_to_agent_state(
        agent_session.agent,
        thread_id=agent_session.thread_id,
        workspace=workspace,
        session=context_session,
        user_message="",
        with_memory=agent_session.with_memory,
        allow_write=enabled,
    )
    return True
