"""从工作区 .llgraph/rules 与 ~/.llgraph/rules 加载 Rule（个人优先，与 Cursor 独立）。"""

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path

from llgraph.session.user_storage import user_rules_dir

_FRONTMATTER_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)$", re.DOTALL)


@dataclass(frozen=True)
class RuleEntry:
    """单条规则。"""

    rule_id: str
    source_path: Path
    description: str
    body: str
    always_apply: bool
    globs: str
    priority: int = 0
    scope: str = "workspace"


def _parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if not value:
        return False
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """
    解析 YAML frontmatter（支持 description: >- 等多行块标量）。

    @param text 文件全文
    @return (meta, body)
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    fm_lines = match.group(1).splitlines()
    meta: dict[str, str] = {}
    i = 0
    while i < len(fm_lines):
        line = fm_lines[i]
        if ":" not in line:
            i += 1
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        block_marker = val.replace(" ", "")
        if block_marker in {">-", ">", "|-", "|"}:
            folded = block_marker.startswith(">")
            i += 1
            block_lines: list[str] = []
            while i < len(fm_lines):
                next_line = fm_lines[i]
                stripped = next_line.strip()
                if not stripped:
                    i += 1
                    continue
                # 下一顶层键（无缩进且含冒号）则结束块
                if (
                    not next_line.startswith((" ", "\t"))
                    and ":" in next_line
                    and not next_line.lstrip().startswith("#")
                ):
                    break
                block_lines.append(stripped)
                i += 1
            if folded:
                meta[key] = " ".join(block_lines).strip()
            else:
                meta[key] = "\n".join(block_lines).strip()
            continue
        meta[key] = val
        i += 1
    return meta, match.group(2).lstrip("\n")


def _read_rule_file(
    path: Path,
    *,
    rules_root: Path,
    rule_id_prefix: str,
    scope: str,
) -> RuleEntry | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    meta, body = _parse_frontmatter(raw)
    rel = path.relative_to(rules_root)
    rule_id = f"{rule_id_prefix}{rel}".replace("\\", "/")
    return RuleEntry(
        rule_id=rule_id,
        source_path=path,
        description=meta.get("description", path.stem),
        body=body.strip(),
        always_apply=_parse_bool(meta.get("alwaysApply")),
        globs=meta.get("globs", "").strip(),
        scope=scope,
    )


def _rule_dedup_key(rule: RuleEntry) -> str:
    """同名文件跨工作区/个人目录时，个人规则覆盖工作区。"""
    return Path(rule.rule_id).name.lower()


def _scan_rules_directory(
    rules_dir: Path,
    *,
    rule_id_prefix: str,
    scope: str,
) -> list[RuleEntry]:
    entries: list[RuleEntry] = []
    if not rules_dir.is_dir():
        return entries
    for path in sorted(rules_dir.rglob("*")):
        if path.suffix.lower() not in (".mdc", ".md", ".txt"):
            continue
        if not path.is_file():
            continue
        rule = _read_rule_file(
            path,
            rules_root=rules_dir,
            rule_id_prefix=rule_id_prefix,
            scope=scope,
        )
        if rule is not None:
            entries.append(rule)
    return entries


LLGRAPH_DIR_NAME = ".llgraph"
LLGRAPH_RULES_DIR_NAME = "rules"


def llgraph_rules_dir(workspace: Path) -> Path:
    """
    llgraph 规则目录（仅此目录，不读取 .cursorrules / .cursor/rules）。

    @param workspace 工作区根
    @return .llgraph/rules 路径
    """
    return workspace / LLGRAPH_DIR_NAME / LLGRAPH_RULES_DIR_NAME


def discover_rules(workspace: Path) -> list[RuleEntry]:
    """
    扫描工作区与个人规则目录；同名文件个人优先。

    @param workspace 工作区根目录
    @return 规则列表（稳定排序）
    """
    ws_root = workspace.expanduser().resolve()
    by_key: dict[str, RuleEntry] = {}
    for rule in _scan_rules_directory(
        llgraph_rules_dir(ws_root),
        rule_id_prefix=".llgraph/rules/",
        scope="workspace",
    ):
        by_key[_rule_dedup_key(rule)] = rule
    for rule in _scan_rules_directory(
        user_rules_dir(),
        rule_id_prefix="user/",
        scope="user",
    ):
        by_key[_rule_dedup_key(rule)] = rule

    entries = list(by_key.values())
    entries.sort(key=lambda r: (-r.priority, r.rule_id))
    return entries


def _glob_hints(glob_pattern: str) -> list[str]:
    """从 glob 提取可用于匹配用户消息的片段。"""
    hints: list[str] = []
    for part in re.split(r"[/\\]+", glob_pattern):
        part = part.strip()
        if not part or part in ("**", "*"):
            continue
        if part.startswith("*."):
            hints.append(part[1:])
            hints.append(part[2:])
        else:
            token = part.replace("*", "")
            if len(token) >= 2:
                hints.append(token.lower())
    return hints


def glob_matches_message(glob_pattern: str, message: str) -> bool:
    """
    判断 glob 规则是否可能与当前用户消息相关。

    @param glob_pattern 如 **/*.java、markdowns/**
    @param message 用户输入
    @return 是否匹配
    """
    if not glob_pattern.strip():
        return False
    msg = message.lower()
    for hint in _glob_hints(glob_pattern):
        if hint in msg:
            return True
    simplified = glob_pattern.replace("**/", "").replace("**", "*")
    if fnmatch.fnmatch(msg, simplified.lower()):
        return True
    return False


def select_rules_for_turn(
    rules: list[RuleEntry],
    *,
    user_message: str,
    session_disabled: set[str],
    session_forced: set[str],
) -> list[RuleEntry]:
    """
    选出本轮应注入的规则（alwaysApply + glob 命中 + 强制 - 禁用）。

    @param rules 全部规则
    @param user_message 用户消息（glob 匹配）
    @param session_disabled 会话禁用 id
    @param session_forced 会话强制启用 id
    @return 生效规则列表
    """
    selected: list[RuleEntry] = []
    for rule in rules:
        if rule.rule_id in session_disabled:
            continue
        if rule.always_apply or rule.rule_id in session_forced:
            selected.append(rule)
            continue
        if not rule.globs:
            continue
        for pattern in re.split(r"[,，\s]+", rule.globs):
            pattern = pattern.strip()
            if pattern and glob_matches_message(pattern, user_message):
                selected.append(rule)
                break
    return selected
