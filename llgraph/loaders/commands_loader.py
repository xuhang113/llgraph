"""自定义扩展命令加载（.llgraph/commands/*.md）。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)$", re.DOTALL)
_COMMANDS_DIR = ".llgraph/commands"


@dataclass(frozen=True)
class CustomCommand:
    """自定义命令定义。"""

    name: str
    description: str
    body: str
    handler: str
    aliases: tuple[str, ...]
    requires_write: bool
    source_path: Path


def _parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if not value:
        return False
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        meta[key.strip()] = val.strip().strip('"').strip("'")
    return meta, match.group(2).lstrip("\n")


def _parse_aliases(raw: str) -> tuple[str, ...]:
    if not raw:
        return ()
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1]
        parts = [p.strip().strip('"').strip("'") for p in inner.split(",")]
        return tuple(p for p in parts if p)
    return tuple(p.strip() for p in raw.split(",") if p.strip())


def discover_commands(workspace: Path) -> list[CustomCommand]:
    """
    扫描 .llgraph/commands/*.md。

    @param workspace 工作区根
    @return 命令列表
    """
    cmd_dir = workspace / _COMMANDS_DIR
    if not cmd_dir.is_dir():
        return []

    commands: list[CustomCommand] = []
    for path in sorted(cmd_dir.glob("*.md")):
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        meta, body = _parse_frontmatter(raw)
        name = meta.get("name", path.stem).strip()
        if not name:
            logger.warning("跳过无效命令文件（缺 name）: %s", path)
            continue
        commands.append(
            CustomCommand(
                name=name,
                description=meta.get("description", name),
                body=body.strip(),
                handler=meta.get("handler", "prompt").strip().lower() or "prompt",
                aliases=_parse_aliases(meta.get("aliases", "")),
                requires_write=_parse_bool(meta.get("requires_write")),
                source_path=path,
            )
        )
    return commands


def resolve_command(workspace: Path, token: str) -> CustomCommand | None:
    """
    按名称或别名查找命令。

    @param workspace 工作区根
    @param token 命令名（不含 /）
    @return CustomCommand 或 None
    """
    key = token.strip().lstrip("/").lower()
    for cmd in discover_commands(workspace):
        if cmd.name.lower() == key:
            return cmd
        if key in (a.lower() for a in cmd.aliases):
            return cmd
    return None


def format_commands_help(workspace: Path) -> str:
    """
    /help 自定义命令区块。

    @param workspace 工作区根
    @return 多行文本
    """
    commands = discover_commands(workspace)
    if not commands:
        return "【自定义命令】（.llgraph/commands/）\n  （无）"
    lines = ["【自定义命令】（.llgraph/commands/）"]
    for cmd in commands:
        alias = f" ({', '.join(cmd.aliases)})" if cmd.aliases else ""
        lines.append(f"  /{cmd.name}{alias}  {cmd.description}")
    return "\n".join(lines)
