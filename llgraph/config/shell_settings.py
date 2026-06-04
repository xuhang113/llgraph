"""Shell 执行配置（.llgraph/agent.json shell 段）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llgraph.config.edit_settings import load_agent_config


@dataclass(frozen=True)
class ShellSettings:
    """终端命令执行参数。"""

    enabled: bool
    timeout_sec: float
    max_output_chars: int
    terminal_log_dir: str
    log_commands: bool


def resolve_shell_settings(workspace: Path) -> ShellSettings:
    """
    解析 shell 配置。

    @param workspace 工作区根
    @return ShellSettings
    """
    cfg = load_agent_config(workspace)
    shell = cfg.get("shell") if isinstance(cfg.get("shell"), dict) else {}

    enabled = shell.get("enabled", True)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() not in ("0", "false", "no")

    try:
        timeout_sec = float(shell.get("timeout_sec", 120))
    except (TypeError, ValueError):
        timeout_sec = 120.0
    timeout_sec = max(5.0, min(timeout_sec, 600.0))

    try:
        max_output_chars = int(shell.get("max_output_chars", 100_000))
    except (TypeError, ValueError):
        max_output_chars = 100_000
    max_output_chars = max(1000, min(max_output_chars, 500_000))

    log_dir = str(shell.get("terminal_log_dir", ".llgraph/context/terminals")).strip()
    if not log_dir:
        log_dir = ".llgraph/context/terminals"

    log_commands = shell.get("log_commands", True)
    if isinstance(log_commands, str):
        log_commands = log_commands.strip().lower() not in ("0", "false", "no")

    return ShellSettings(
        enabled=bool(enabled),
        timeout_sec=timeout_sec,
        max_output_chars=max_output_chars,
        terminal_log_dir=log_dir,
        log_commands=bool(log_commands),
    )
