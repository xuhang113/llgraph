"""从工作区 .llgraph/thought 加载可配置的 Agent 规划/检索规范（Thought）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from llgraph.core.agent_config import load_agent_config
from llgraph.loaders.prompt_loader import compose_thought_block_header, compose_thought_builtin_retrieval
from llgraph.loaders.rules_loader import _parse_bool, _parse_frontmatter

LLGRAPH_DIR_NAME = ".llgraph"
THOUGHT_DIR_NAME = "thought"
AGENT_CONFIG_FILE = "agent.json"

@dataclass(frozen=True)
class ThoughtEntry:
    """单条 Thought 规范。"""

    thought_id: str
    source_path: Path
    description: str
    body: str
    enabled: bool
    priority: int


@dataclass(frozen=True)
class ThoughtSettings:
    """agent.json 中 thought 段配置。"""

    enabled: bool
    max_chars: int
    emit_plan_line: bool
    use_builtin_fallback: bool


def _default_thought_settings() -> ThoughtSettings:
    return ThoughtSettings(
        enabled=True,
        max_chars=8_000,
        emit_plan_line=True,
        use_builtin_fallback=True,
    )


def llgraph_thought_dir(workspace: Path) -> Path:
    """
    Thought 规范目录。

    @param workspace 工作区根
    @return .llgraph/thought 路径
    """
    return workspace / LLGRAPH_DIR_NAME / THOUGHT_DIR_NAME


def agent_config_path(workspace: Path) -> Path:
    """工作区 agent.json 路径。"""
    return workspace / LLGRAPH_DIR_NAME / AGENT_CONFIG_FILE


def load_thought_settings(workspace: Path) -> ThoughtSettings:
    """
    读取 agent.json 中的 thought 配置（用户 + 工作区合并）。

    @param workspace 工作区根
    @return ThoughtSettings
    """
    defaults = _default_thought_settings()
    raw = load_agent_config(workspace)

    section = raw.get("thought")
    if not isinstance(section, dict):
        return defaults

    max_chars = section.get("max_chars", defaults.max_chars)
    if isinstance(max_chars, str) and max_chars.isdigit():
        max_chars = int(max_chars)
    if not isinstance(max_chars, int) or max_chars < 500:
        max_chars = defaults.max_chars

    return ThoughtSettings(
        enabled=_parse_bool(section.get("enabled", defaults.enabled)),
        max_chars=max_chars,
        emit_plan_line=_parse_bool(
            section.get("emit_plan_line", defaults.emit_plan_line)
        ),
        use_builtin_fallback=_parse_bool(
            section.get("use_builtin_fallback", defaults.use_builtin_fallback)
        ),
    )


def _read_thought_file(path: Path, workspace: Path) -> ThoughtEntry | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None

    meta, body = _parse_frontmatter(raw)
    rel = path.relative_to(workspace)
    thought_id = str(rel).replace("\\", "/")

    priority_raw = meta.get("priority", "0")
    try:
        priority = int(priority_raw)
    except (TypeError, ValueError):
        priority = 0

    enabled = _parse_bool(meta.get("enabled", True))

    return ThoughtEntry(
        thought_id=thought_id,
        source_path=path,
        description=meta.get("description", path.stem),
        body=body.strip(),
        enabled=enabled,
        priority=priority,
    )


def discover_thoughts(workspace: Path) -> list[ThoughtEntry]:
    """
    扫描 .llgraph/thought 下 .md / .mdc 文件。

    @param workspace 工作区根
    @return 按 priority 降序排列的条目
    """
    root = llgraph_thought_dir(workspace)
    if not root.is_dir():
        return []

    entries: list[ThoughtEntry] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".md", ".mdc"):
            continue
        entry = _read_thought_file(path, workspace)
        if entry is not None and entry.enabled and entry.body:
            entries.append(entry)

    entries.sort(key=lambda item: (-item.priority, item.thought_id))
    return entries


def build_thought_prompt_block(workspace: Path) -> str:
    """
    组装注入 system prompt 的 Thought 规范块。

    @param workspace 工作区根
    @return 空字符串表示不注入
    """
    settings = load_thought_settings(workspace)
    if not settings.enabled:
        return ""

    parts: list[str] = []
    header = compose_thought_block_header(emit_plan_line=settings.emit_plan_line)
    if header:
        parts.extend([header, ""])

    entries = discover_thoughts(workspace)
    if entries:
        for entry in entries:
            parts.append(f"### {entry.description} (`{entry.thought_id}`)")
            parts.append(entry.body)
            parts.append("")
    elif settings.use_builtin_fallback:
        parts.append(compose_thought_builtin_retrieval())
        parts.append("")

    if len(parts) <= 2:
        return ""

    text = "\n".join(parts).strip()
    if len(text) > settings.max_chars:
        text = text[: settings.max_chars - 40] + "\n\n…（Thought 规范已截断）"
    return text


def thought_summary(workspace: Path) -> str:
    """
    会话横幅用的一行摘要。

    @param workspace 工作区根
    @return 摘要文本
    """
    settings = load_thought_settings(workspace)
    if not settings.enabled:
        return "已关闭（agent.json thought.enabled=false）"

    entries = discover_thoughts(workspace)
    if entries:
        names = ", ".join(e.description for e in entries[:3])
        suffix = f" 等{len(entries)}条" if len(entries) > 3 else ""
        plan = "需【规划】前缀" if settings.emit_plan_line else "无强制规划行"
        return f"{len(entries)} 条（{names}{suffix}；{plan}）"

    if settings.use_builtin_fallback:
        plan = "需【规划】前缀" if settings.emit_plan_line else "无强制规划行"
        return f"内置默认检索重试规范（{plan}）"
    return "无（可添加 .llgraph/thought/*.md）"
