"""从工作区 .llgraph/thought 加载可配置的 Agent 规划/检索规范（Thought）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from llgraph.core.agent_config import load_agent_config
from llgraph.loaders.rules_loader import _parse_bool, _parse_frontmatter

LLGRAPH_DIR_NAME = ".llgraph"
THOUGHT_DIR_NAME = "thought"
AGENT_CONFIG_FILE = "agent.json"

# 包内默认 Thought（工作区未配置时使用）
_BUILTIN_RETRIEVAL_THOUGHT = """\
## 调工具前的规划（Thought）

每轮**准备调用工具之前**，先用 1～3 句中文说明：当前目标、上一步结果、本步打算做什么。
若终端为 `/trace all`，这段规划会显示在「模型规划」之前，便于对照。

## 目录浏览与文件发现（禁止 shell 替代）

- **list_directory**：列目录（如 `path="docs"`、`.llgraph/context/tool-results`）；**禁止** `ls` / `ls -la`。
- **glob_files**：按文件名找（如 `**/collect_alert.sh`）；**禁止** `find`。
- **grep_files**：按内容搜（含 .md）；**禁止** shell `grep`/`rg`。
- **read_file**：读已知路径文件；**禁止** shell `cat`/`head`/`tail`。

## 批量文件名检索（禁止逐个 glob）

- **多个已知文件名/脚本名**：**一次** `grep_files`（如 `pattern="collect_alert|gitlab_monitor|dm-daily-task"`，`path="markdowns"` 或 `path="."`）；**禁止**对每个名字单独 `glob_files`。
- **glob 全未命中或连续多个 glob 为 0**：源文件可能不在工作区，仅在 **markdowns/docs** 台账/crontab 文档中被引用；改 `grep_files`（`path="markdowns"` 或 `path="docs"`）或 `read_file` 已知文档，**勿再逐个 glob**。

## 检索无结果时必须重试（勿立刻断言「不存在」）

1. **glob_files**：文件名是否存在（如 `**/collect_alert.sh`）。
2. **grep_files**：字面/正则搜内容（含 .md）；多词用 `词A|词B`。
3. **search_code_parallel**：当轮不明确时可选一次；query 里自行扩展关键字；之后 grep_files 深挖。
4. **search_workspace**：`keywords` 一次 **5～12** 个词；换 `path`（`markdowns`、`docs`）。
5. **read_file**：manifest、embedding.json、README 等配置先读后答。

同一用户问题内，至少换 **2 种**工具；**禁止**对同一 find/glob/ls 空结果重复超过 1 次。
"""


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

    parts: list[str] = ["## Agent 规划与检索规范（Thought，工作区可配置）", ""]

    if settings.emit_plan_line:
        parts.append(
            "硬性要求：每次调用工具之前，必须先输出 1～3 句中文规划，"
            "以「【规划】」开头，说明目标与上一步结果；然后再发起 tool_calls。"
        )
        parts.append("")

    entries = discover_thoughts(workspace)
    if entries:
        for entry in entries:
            parts.append(f"### {entry.description} (`{entry.thought_id}`)")
            parts.append(entry.body)
            parts.append("")
    elif settings.use_builtin_fallback:
        parts.append(_BUILTIN_RETRIEVAL_THOUGHT.strip())
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
