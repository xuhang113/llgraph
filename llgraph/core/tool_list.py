"""Agent 工具列表（/tools 元命令）。"""

from __future__ import annotations

from pathlib import Path

from llgraph.core.tools import get_agent_tools

_MCP_NAME_PREFIX = "mcp__"

_BUILTIN_GROUPS: tuple[tuple[str, frozenset[str]], ...] = (
    ("通用", frozenset({"get_current_utc_time"})),
    (
        "文件",
        frozenset({
            "list_directory",
            "search_workspace",
            "search_files",
            "grep_files",
            "read_file",
            "search_replace",
            "append_file",
            "write_file",
        }),
    ),
    ("Shell", frozenset({"run_shell_command"})),
    (
        "代码索引",
        frozenset({"search_code_semantic", "search_code_hybrid"}),
    ),
    ("Web", frozenset({"web_search"})),
)


def _tool_name(tool: object) -> str:
    """
    解析工具名。

    @param tool LangChain Tool
    @return 工具名
    """
    name = getattr(tool, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return str(tool)


def _tool_description_one_line(tool: object, *, max_len: int = 96) -> str:
    """
    工具描述首行摘要。

    @param tool LangChain Tool
    @param max_len 最大长度
    @return 单行描述
    """
    desc = getattr(tool, "description", None) or ""
    if not isinstance(desc, str):
        desc = str(desc)
    line = desc.strip().splitlines()[0] if desc.strip() else "（无描述）"
    if len(line) > max_len:
        return line[: max_len - 3] + "..."
    return line


def _is_mcp_tool(name: str) -> bool:
    """
    是否为 MCP 工具。

    @param name 工具名
    @return 是否 MCP
    """
    return name.startswith(_MCP_NAME_PREFIX)


def _group_label(name: str) -> str:
    """
    内置工具分组标签。

    @param name 工具名
    @return 分组名
    """
    for label, names in _BUILTIN_GROUPS:
        if name in names:
            return label
    return "其他"


def format_agent_tools_report(
    workspace: Path,
    *,
    allow_write: bool = False,
    web_search_enabled: bool = False,
    mcp_tools: list | None = None,
) -> str:
    """
    格式化当前会话可用工具列表。

    @param workspace 工作区根
    @param allow_write 是否可写
    @param web_search_enabled 是否启用 web_search
    @param mcp_tools 已加载 MCP 工具
    @return 多行文本
    """
    tools = get_agent_tools(
        workspace_root=workspace,
        allow_write=allow_write,
        mcp_tools=mcp_tools,
        web_search_enabled=web_search_enabled,
    )

    builtin: list[tuple[str, str, str]] = []
    mcp: list[tuple[str, str, str]] = []
    for tool in tools:
        name = _tool_name(tool)
        desc = _tool_description_one_line(tool)
        if _is_mcp_tool(name):
            mcp.append((name, desc, _mcp_server_from_name(name)))
        else:
            builtin.append((name, desc, _group_label(name)))

    write_state = "可写" if allow_write else "只读"
    web_state = "已启用" if web_search_enabled else "未启用（/web on）"

    lines = [
        "Agent 工具",
        "========",
        f"文件模式: {write_state}  |  Web 搜索: {web_state}",
        f"内置: {len(builtin)}  |  MCP: {len(mcp)}  |  合计: {len(tools)}",
        "",
        f"内置工具 ({len(builtin)})",
        "----------",
    ]

    if not builtin:
        lines.append("  （无）")
    else:
        by_group: dict[str, list[tuple[str, str]]] = {}
        for name, desc, group in builtin:
            by_group.setdefault(group, []).append((name, desc))
        group_order = [g for g, _ in _BUILTIN_GROUPS] + ["其他"]
        seen_groups: set[str] = set()
        for group in group_order:
            if group not in by_group:
                continue
            seen_groups.add(group)
            lines.append(f"  [{group}]")
            for name, desc in sorted(by_group[group]):
                lines.append(f"    {name}")
                lines.append(f"      {desc}")
        for group, items in sorted(by_group.items()):
            if group in seen_groups:
                continue
            lines.append(f"  [{group}]")
            for name, desc in sorted(items):
                lines.append(f"    {name}")
                lines.append(f"      {desc}")

    lines.extend(["", f"MCP 工具 ({len(mcp)})", "----------"])
    if not mcp:
        lines.append("  （未加载；配置 .llgraph/mcp.json）")
    else:
        by_server: dict[str, list[tuple[str, str]]] = {}
        for name, desc, server in mcp:
            by_server.setdefault(server, []).append((name, desc))
        for server in sorted(by_server):
            lines.append(f"  [{server}]")
            for name, desc in sorted(by_server[server]):
                short = name[len(f"mcp__{server}__") :]
                lines.append(f"    {short}  ({name})")
                lines.append(f"      {desc}")

    lines.extend(
        [
            "",
            "说明: 列表按当前会话状态生成（/write、/web 切换后需重新 /tools）。",
            "      内置工具由 llgraph 代码注册；MCP 来自 .llgraph/mcp.json。",
        ]
    )
    return "\n".join(lines)


def _mcp_server_from_name(name: str) -> str:
    """
    从 mcp__server__tool 解析 server 名。

    @param name 完整工具名
    @return server 名
    """
    if not name.startswith(_MCP_NAME_PREFIX):
        return "unknown"
    rest = name[len(_MCP_NAME_PREFIX) :]
    parts = rest.split("__", 1)
    return parts[0] if parts else "unknown"
