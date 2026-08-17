"""工作区能力清单：工具 / MCP / Skill / Rule。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llgraph.core.tools import get_agent_tools
from llgraph.loaders.commands_loader import discover_commands
from llgraph.loaders.rules_loader import discover_rules
from llgraph.loaders.skills_loader import discover_skills
from llgraph.config.mcp_config import resolve_mcp_settings


def _tool_info(tool: Any) -> dict[str, Any]:
    name = getattr(tool, "name", None) or str(tool)
    desc = getattr(tool, "description", "") or ""
    return {"name": name, "description": desc}


def load_capabilities(workspace: Path, *, allow_write: bool = False) -> dict[str, Any]:
    """
    加载工作区可用能力。

    @param workspace 工作区根
    @param allow_write 是否包含写工具
    @return 能力 dict
    """
    from llgraph.console.runtime.workspace_runtime import RUNTIME_MANAGER

    rt = RUNTIME_MANAGER.get(workspace, allow_write=allow_write)
    active_skills_lower = {s.lower() for s in rt.context_session.active_skills}
    builtin = get_agent_tools(
        workspace_root=workspace,
        allow_write=allow_write,
        mcp_tools=[],
        web_search_enabled=rt.web_search_enabled,
        sandbox_policy=rt.sandbox_policy,
    )
    mcp_cfg = resolve_mcp_settings(workspace)
    from llgraph.config.catalog_paths import format_catalog_path, scope_label

    skills = discover_skills(workspace)
    rules = discover_rules(workspace)
    commands = discover_commands(workspace)

    load_errors = list(getattr(rt.mcp_registry, "load_errors", []) or [])
    error_by_name: dict[str, str] = {}
    for err in load_errors:
        if ":" in err:
            name, _, rest = err.partition(":")
            error_by_name[name.strip()] = rest.strip()
        else:
            error_by_name[err] = err
    active_servers = set()
    if rt.mcp_registry is not None:
        active_servers = set(getattr(rt.mcp_registry, "_runtimes", {}) or {})
    tool_count_by_server: dict[str, int] = {}
    for t in rt.mcp_tools:
        tname = getattr(t, "name", "") or ""
        # mcp__<server>__<tool>
        parts = tname.split("__")
        if len(parts) >= 3 and parts[0] == "mcp":
            srv = parts[1]
            tool_count_by_server[srv] = tool_count_by_server.get(srv, 0) + 1

    mcp_servers_out: list[dict[str, Any]] = []
    for s in mcp_cfg.servers if mcp_cfg else []:
        if s.name in active_servers:
            status = "ok"
        elif s.name in error_by_name:
            status = "error"
        elif not getattr(rt, "mcp_ready", None) or not rt.mcp_ready.is_set():
            status = "loading"
        else:
            status = "error" if load_errors else "idle"
        mcp_servers_out.append(
            {
                "name": s.name,
                "command": s.command,
                "enabled": s.enabled,
                "status": status,
                "error": error_by_name.get(s.name),
                "tool_count": tool_count_by_server.get(s.name, 0),
            }
        )

    return {
        "builtin_tools": [_tool_info(t) for t in builtin],
        "mcp_tools": [_tool_info(t) for t in rt.mcp_tools],
        "mcp_summary": rt.mcp_summary,
        "mcp_errors": load_errors,
        "mcp_loading": not (getattr(rt, "mcp_ready", None) and rt.mcp_ready.is_set()),
        "mcp_servers": mcp_servers_out,
        "skills": [
            {
                "name": s.name,
                "description": s.description,
                "scope": s.scope,
                "scope_label": scope_label(s.scope),
                "path": format_catalog_path(
                    workspace, s.skill_dir / "SKILL.md", s.scope
                ),
                "active": s.name.lower() in active_skills_lower,
            }
            for s in skills
        ],
        "rules": [
            {
                "id": r.rule_id,
                "description": r.description,
                "scope": r.scope,
                "scope_label": scope_label(r.scope),
                "path": format_catalog_path(workspace, r.source_path, r.scope),
                "forced": r.rule_id in rt.context_session.forced_rules,
                "disabled": r.rule_id in rt.context_session.disabled_rules,
            }
            for r in rules
        ],
        "context_state": {
            "active_skills": list(rt.context_session.active_skills),
            "disabled_rules": sorted(rt.context_session.disabled_rules),
            "forced_rules": sorted(rt.context_session.forced_rules),
        },
        "commands": [
            {"name": c.name, "description": c.description, "requires_write": c.requires_write}
            for c in commands
        ],
        "web_search_enabled": rt.web_search_enabled,
        "trace_mode": rt.trace_session.mode.value,
    }
