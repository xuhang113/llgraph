"""会话内 Web 搜索开关（/web on|off）。"""

from __future__ import annotations

from llgraph.core.agent import rebuild_agent_preserving_memory
from llgraph.core.agent_session import AgentSessionContext
from llgraph.config.web_search_settings import (
    resolve_web_search_settings,
    validate_web_search_ready,
)


def format_web_search_status(agent_session: AgentSessionContext) -> str:
    """
    当前 Web 搜索模式说明。

    @param agent_session Agent 会话
    @return 多行文本
    """
    workspace = agent_session.workspace
    settings = resolve_web_search_settings(workspace)
    if agent_session.web_search_enabled:
        mode = "已启用（web_search 工具已注册）"
    else:
        mode = "已禁用"
    lines = [
        f"Web 搜索: {mode}",
        f"配置: ~/.llgraph/agent.json → web_search（depth={settings.search_depth}, "
        f"max_results={settings.max_results}, timeout={settings.timeout_sec:g}s）",
        "命令: /web on  |  /web off",
        f"凭据: 环境变量 {settings.api_key_env}（~/.config/llgraph/llgraph.env）",
    ]
    if not agent_session.web_search_enabled:
        ok, err = validate_web_search_ready(workspace)
        if not ok:
            lines.append(f"启用前需: {err}")
    return "\n".join(lines)


def set_session_web_search_mode(
    agent_session: AgentSessionContext,
    *,
    enabled: bool,
) -> tuple[bool, str]:
    """
    切换 Web 搜索并重建 Agent（保留对话历史）。

    @param agent_session Agent 会话
    @param enabled 是否启用
    @return (是否发生切换, 提示信息)
    """
    if agent_session.web_search_enabled == enabled:
        if enabled:
            return False, "当前 Web 搜索已启用。"
        return False, "当前 Web 搜索已禁用。"

    if enabled:
        ok, err = validate_web_search_ready(agent_session.workspace)
        if not ok:
            return False, err

    write_mode = agent_session.allow_write
    rebuild_agent_preserving_memory(
        agent_session,
        allow_write=write_mode,
        web_search_enabled=enabled,
        mcp_tools=agent_session.mcp_tools,
        on_file_changed=agent_session.on_file_changed if write_mode else None,
    )
    agent_session.web_search_enabled = enabled
    if enabled:
        return True, "已启用 Web 搜索（web_search 工具已注册，会话历史已保留）。"
    return True, "已禁用 Web 搜索（web_search 工具已移除）。"


def resolve_initial_web_search_enabled(workspace) -> bool:
    """
    会话启动时是否默认启用 Web 搜索。

    @param workspace 工作区根
    @return 是否启用
    """
    from pathlib import Path

    root = Path(workspace).expanduser().resolve()
    settings = resolve_web_search_settings(root)
    if not settings.default_enabled:
        return False
    ok, _ = validate_web_search_ready(root)
    return ok
