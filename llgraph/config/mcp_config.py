"""MCP Server 配置加载（仅 .llgraph/mcp.json）。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

LLGRAPH_MCP_FILENAME = "mcp.json"


@dataclass(frozen=True)
class McpServerConfig:
    """单个 MCP Server 配置。"""

    name: str
    command: str
    args: list[str]
    env: dict[str, str]
    cwd: str | None
    enabled: bool


@dataclass(frozen=True)
class McpSettings:
    """MCP 全局配置。"""

    servers: tuple[McpServerConfig, ...]
    timeout_sec: float
    allow_write_tools: bool
    config_source: str


def _parse_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _expand_placeholders(value: str, workspace: Path) -> str:
    """
    展开 ${WORKSPACE_ROOT} / ${LLGRAPH_WORKSPACE} 与系统环境变量。

    @param value 原始字符串
    @param workspace 工作区根
    @return 展开后字符串
    """
    root = str(workspace.expanduser().resolve())
    out = (
        value.replace("${WORKSPACE_ROOT}", root)
        .replace("$WORKSPACE_ROOT", root)
        .replace("${LLGRAPH_WORKSPACE}", root)
        .replace("$LLGRAPH_WORKSPACE", root)
    )
    return os.path.expandvars(out)


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _servers_from_llgraph_format(
    raw: dict, workspace: Path
) -> tuple[McpServerConfig, ...]:
    """从 .llgraph/mcp.json 的 servers 段解析。"""
    servers_raw = raw.get("servers")
    if not isinstance(servers_raw, dict):
        return ()
    disabled = raw.get("disabled_servers") or []
    if isinstance(disabled, str):
        disabled = [disabled]
    disabled_set = {str(x) for x in disabled} if isinstance(disabled, list) else set()

    servers: list[McpServerConfig] = []
    for name, cfg in servers_raw.items():
        if not isinstance(cfg, dict):
            continue
        command = str(cfg.get("command", "")).strip()
        if not command:
            continue
        args = cfg.get("args") or []
        if not isinstance(args, list):
            args = []
        env_raw = cfg.get("env") or {}
        env = (
            {
                str(k): _expand_placeholders(str(v), workspace)
                for k, v in env_raw.items()
            }
            if isinstance(env_raw, dict)
            else {}
        )
        # 子进程可读取工作区根
        env.setdefault("WORKSPACE_ROOT", str(workspace.expanduser().resolve()))
        env.setdefault("LLGRAPH_WORKSPACE", str(workspace.expanduser().resolve()))
        cwd = cfg.get("cwd")
        cwd_str = _expand_placeholders(str(cwd), workspace) if cwd else None
        enabled = _parse_bool(cfg.get("enabled"), True) and name not in disabled_set
        servers.append(
            McpServerConfig(
                name=name,
                command=_expand_placeholders(command, workspace),
                args=[_expand_placeholders(str(a), workspace) for a in args],
                env=env,
                cwd=cwd_str,
                enabled=enabled,
            )
        )
    return tuple(servers)


def resolve_mcp_settings(workspace: Path, *, allow_write: bool = False) -> McpSettings:
    """
    解析 MCP 配置（仅读取工作区 .llgraph/mcp.json）。

    @param workspace 工作区根
    @param allow_write Agent 是否 -w 模式
    @return McpSettings
    """
    workspace = workspace.expanduser().resolve()
    llgraph_path = workspace / ".llgraph" / LLGRAPH_MCP_FILENAME

    defaults_timeout = 60.0
    allow_write_tools = False
    servers: tuple[McpServerConfig, ...] = ()
    source = "(无)"

    if llgraph_path.is_file():
        raw = _load_json(llgraph_path)
        defaults = raw.get("defaults") if isinstance(raw.get("defaults"), dict) else {}
        try:
            defaults_timeout = float(defaults.get("timeout_sec", defaults_timeout))
        except (TypeError, ValueError):
            pass
        allow_write_tools = _parse_bool(defaults.get("allow_write_tools"), False)
        servers = _servers_from_llgraph_format(raw, workspace)
        source = str(llgraph_path)

    if not allow_write:
        allow_write_tools = False

    enabled = tuple(s for s in servers if s.enabled)
    return McpSettings(
        servers=enabled,
        timeout_sec=max(5.0, defaults_timeout),
        allow_write_tools=allow_write_tools,
        config_source=source,
    )


def format_mcp_summary(settings: McpSettings) -> str:
    """
    /help 用 MCP 摘要。

    @param settings MCP 配置
    @return 多行文本
    """
    if not settings.servers:
        return "MCP: 未加载（在 .llgraph/mcp.json 的 servers 中配置）"
    lines = [f"MCP: {len(settings.servers)} 个 Server（{settings.config_source}）"]
    for srv in settings.servers:
        lines.append(f"  - {srv.name}")
    return "\n".join(lines)
