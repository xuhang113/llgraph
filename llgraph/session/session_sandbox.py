"""会话内 OS 沙箱开关（Web / CLI 覆盖 sandbox.json）。"""

from __future__ import annotations

from llgraph.config.sandbox_settings import resolve_sandbox_settings
from llgraph.core.agent import rebuild_agent_preserving_memory
from llgraph.core.agent_session import AgentSessionContext
from llgraph.sandbox.policy import build_sandbox_policy


def format_sandbox_status(agent_session: AgentSessionContext) -> str:
    """
    当前 OS 沙箱模式说明。

    @param agent_session Agent 会话
    @return 多行文本
    """
    policy = agent_session.sandbox_policy
    cli = getattr(agent_session, "sandbox_cli_enabled", None)
    if policy is not None and policy.enabled:
        mode = f"已启用（{policy.backend}，mode={policy.mode}，network={policy.network}）"
    elif policy is not None and policy.active:
        mode = "已请求启用但后端不可用"
    else:
        mode = "未启用"
    lines = [
        f"OS 沙箱: {mode}",
        "Web 顶栏可切换；CLI 可用 --sandbox / --no-sandbox 覆盖 sandbox.json。",
    ]
    if cli is True:
        lines.append("当前覆盖: 强制开启")
    elif cli is False:
        lines.append("当前覆盖: 强制关闭")
    if policy is not None and policy.startup_warning():
        lines.append(f"注意: {policy.startup_warning()}")
    return "\n".join(lines)


def set_session_sandbox_mode(
    agent_session: AgentSessionContext,
    *,
    enabled: bool,
) -> tuple[bool, str]:
    """
    切换 OS 沙箱并重建 Agent（保留对话历史）。

    @param agent_session Agent 会话
    @param enabled 是否启用
    @return (是否发生切换, 提示信息)
    """
    policy = agent_session.sandbox_policy
    currently_enabled = policy is not None and policy.enabled
    if currently_enabled == enabled:
        if enabled:
            return False, "当前 OS 沙箱已启用。"
        return False, "当前 OS 沙箱已禁用。"

    workspace = agent_session.workspace
    sandbox_settings = resolve_sandbox_settings(workspace)
    cli_enabled = True if enabled else False
    new_policy = build_sandbox_policy(
        workspace,
        sandbox_settings,
        cli_enabled=cli_enabled,
        allow_write=agent_session.allow_write,
    )
    if enabled and not new_policy.enabled:
        warning = new_policy.startup_warning()
        return False, warning or "沙箱后端不可用，无法启用。"

    write_mode = agent_session.allow_write
    rebuild_agent_preserving_memory(
        agent_session,
        allow_write=write_mode,
        mcp_tools=agent_session.mcp_tools,
        on_file_changed=agent_session.on_file_changed if write_mode else None,
        sandbox_policy=new_policy,
    )
    agent_session.sandbox_policy = new_policy
    agent_session.sandbox_cli_enabled = cli_enabled
    if enabled:
        return True, (
            f"已启用 OS 沙箱（{new_policy.backend}，mode={new_policy.mode}，"
            "会话历史已保留）。"
        )
    return True, "已禁用 OS 沙箱（会话历史已保留）。"
