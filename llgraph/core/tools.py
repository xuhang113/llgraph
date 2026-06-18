"""Agent 工具注册：通用工具 + 工作区文件工具。"""

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

from llgraph.code_index.index_settings import resolve_index_settings
from llgraph.core.code_index_tools import create_code_index_tools
from llgraph.context.context_spill import ContextSpill, apply_spill_to_tools
from llgraph.core.filesystem_tools import create_filesystem_tools
from llgraph.session.session_history_tools import create_session_history_tools
from llgraph.core.shell_tools import create_shell_tools
from llgraph.core.web_search_tools import create_web_search_tools
from llgraph.plan.plan_tools import create_plan_tools
from llgraph.config.mcp_config import resolve_mcp_settings
from llgraph.core.mcp_tools import McpToolRegistry, create_mcp_tools
from llgraph.survey.edit_confirm import EditConfirmGate
from llgraph.session.session_edits import SessionEditTracker
from llgraph.core.write_failure_tracker import WriteFailureTracker
from llgraph.core.workspace import WorkspaceContext
from llgraph.sandbox.policy import SandboxPolicy


@tool
def get_current_utc_time() -> str:
    """返回当前 UTC 时间（ISO 8601），用于回答「现在几点」类问题。"""
    return datetime.now(timezone.utc).isoformat()


def get_agent_tools(
    *,
    workspace_root: str | Path | None = None,
    allow_write: bool = False,
    edit_tracker: SessionEditTracker | None = None,
    on_file_changed: Callable[[str], None] | None = None,
    mcp_tools: list | None = None,
    context_spill: ContextSpill | None = None,
    write_failure_tracker: WriteFailureTracker | None = None,
    web_search_enabled: bool = False,
    edit_confirm_gate: EditConfirmGate | None = None,
    sandbox_policy: SandboxPolicy | None = None,
) -> list:
    """
    组装 Agent 可用工具列表。

    @param workspace_root 工作区根目录，默认当前进程 cwd
    @param allow_write 是否注册 write_file / search_replace 等（默认 False，只读）
    @param edit_tracker 会话编辑账本
    @param on_file_changed 写文件成功后的回调（相对路径）
    @param mcp_tools 已加载的 MCP 工具
    @param context_spill 工具结果落盘（P6）
    @param write_failure_tracker 写工具失败计数
    @param web_search_enabled 是否注册 web_search（Tavily）
    @param sandbox_policy OS 沙箱策略；None 时按 sandbox.json 默认（通常关闭）
    @return Tool 列表
    """
    root = Path(workspace_root or ".").expanduser().resolve()
    skip_dirs = frozenset(resolve_index_settings(root).skip_dirs)
    ctx = WorkspaceContext(
        root,
        allow_write=allow_write,
        extra_skip_dirs=skip_dirs,
        sandbox_policy=sandbox_policy,
    )
    fs_tools = create_filesystem_tools(
        ctx,
        edit_tracker=edit_tracker,
        on_file_changed=on_file_changed,
        write_failure_tracker=write_failure_tracker,
        edit_confirm_gate=edit_confirm_gate,
    )
    index_tools = create_code_index_tools(root)
    history_tools = create_session_history_tools(root)
    plan_tools = create_plan_tools(root)
    shell_tools = create_shell_tools(ctx, allow_write=allow_write)
    sandbox_blocks_network = (
        sandbox_policy is not None
        and sandbox_policy.enabled
        and sandbox_policy.network == "deny"
    )
    web_tools = (
        create_web_search_tools(root)
        if web_search_enabled and not sandbox_blocks_network
        else []
    )
    extra = list(mcp_tools or [])
    tools = [
        get_current_utc_time,
        *fs_tools,
        *shell_tools,
        *index_tools,
        *history_tools,
        *plan_tools,
        *web_tools,
        *extra,
    ]
    return apply_spill_to_tools(tools, context_spill)


def load_mcp_tool_bundle(
    workspace: Path,
    *,
    allow_write: bool = False,
) -> tuple[list, McpToolRegistry | None, str]:
    """
    加载 MCP 工具与 registry。

    @param workspace 工作区根
    @param allow_write 是否允许 MCP 写类工具
    @return (tools, registry, summary)
    """
    settings = resolve_mcp_settings(workspace, allow_write=allow_write)
    tools, registry = create_mcp_tools(settings)
    from llgraph.config.mcp_config import format_mcp_summary

    summary = format_mcp_summary(settings)
    if registry is not None:
        summary = registry.summary()
    return tools, registry, summary
