"""MCP 工具读写分类：只读模式下该隐藏哪些外部工具。

只读是 llgraph 的默认模式，这一层判错两头都疼：

- **判成读（漏放）**：模型在只读会话里拿到了真正会改东西的工具。
  实测 `mcp-server-git` 的 `git_commit` / `git_add` / `git_reset` / `git_checkout`
  与 filesystem 的 `move_file` 全都漏放过——`git_reset` / `git_checkout`
  会直接冲掉用户未提交的改动。
- **判成写（误伤）**：只读工具被悄悄摘掉，模型看不见它、用户也不知道少了什么，
  唯一的补救是把整台 Server 的写工具全打开。

所以判定按可靠性分层，先用服务端自己声明的标注，再退回命名约定，
最后才看描述的**起始动词**——整段描述做子串匹配是最不可靠的一档
（`progress updates` 里的 `update`、`get_pull_request` 里的 `pull` 都会误命中），
本模块不再这么做。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fnmatch import fnmatchcase

SOURCE_OVERRIDE = "override"
SOURCE_ANNOTATION = "annotation"
SOURCE_NAME = "name"
SOURCE_DESCRIPTION = "description"
SOURCE_DEFAULT = "default"

# 命名里出现即判写：都是「只可能在改东西」的动词。
# 收录标准是「想不出哪个只读工具会用它当动词」——例如 `pull` 就不收，
# 因为 `get_pull_request` / `list_pull_requests` 这类只读工具全带这个词。
_HARD_WRITE_VERBS = frozenset({
    "add", "alter", "append", "apply", "approve", "archive", "assign",
    "ban", "cancel", "checkout", "chmod", "chown", "clear", "clone", "close",
    "comment", "commit", "compress", "copy", "cp", "create", "del", "delete",
    "deploy", "destroy", "disable", "drop", "edit", "enable", "flush", "follow",
    "fork", "grant", "gzip", "import", "init", "insert", "install", "invite",
    "kick", "kill", "lock", "merge", "mkdir", "modify", "move", "mute", "mv",
    "overwrite", "patch", "persist", "pin", "post", "prepend", "publish", "purge",
    "push", "put", "rebase", "reboot", "register", "reject", "remove", "rename",
    "reopen", "replace", "reset", "restart", "restore", "revert", "revoke",
    "react", "reply", "rerun", "retry", "rm", "rmdir", "rollback", "save",
    "send", "set", "stage", "star", "stash",
    "store", "submit", "subscribe", "switch", "sync", "terminate", "touch",
    "toggle", "transfer", "truncate", "unarchive", "unassign", "unfollow",
    "uninstall", "unlock", "unpin", "unregister", "unstage", "unstar",
    "unsubscribe", "update", "upload", "upsert", "wipe", "write",
})

# 只在「动词位」出现且同位置没有读动词时才判写：
# 这些词本身不表示改动（`run_query` / `execute_sql` 可能只是查），
# 但放在只读模式里默认按写处理更安全。
_SOFT_WRITE_VERBS = frozenset({
    "call", "click", "eval", "evaluate", "exec", "execute", "fill", "invoke",
    "manage", "press", "request", "run", "start", "stop", "type",
})

# 读动词：既用来否决上面的软动词，也用来在扫描全名之前短路。
# 不收 `open` / `me` / `is` 这类语义太宽的词——它们交给「默认按读」兜底，
# 收进来只会让 `open_issue` 这种写工具被短路成读。
_READ_VERBS = frozenset({
    "analyse", "analyze", "browse", "cat", "check", "columns", "compare",
    "convert", "count", "current", "describe", "diff", "docs", "echo", "explain",
    "fetch", "find", "get", "grep", "head", "history", "info", "inspect", "list",
    "log", "logs", "lookup", "ls", "ping", "preview", "query", "read", "report",
    "schema", "screenshot", "search", "select", "show", "snapshot", "stat",
    "stats", "status", "summarize", "summary", "tables", "tail", "tree", "view",
})

# 描述起始动词专属：不会出现在工具名里，但描述常用它们开头
# （`git_commit` 的描述就是 "Records changes to the repository"）。
_DESCRIPTION_WRITE_VERBS = _HARD_WRITE_VERBS | frozenset({
    "adjust", "mark", "mutate", "record", "rewrite", "stage", "unstage", "upsert",
})

# 工具名里的动词位：MCP 命名约定是动词打头，但常带一层 Server 前缀
# （`git_commit`、`slack_post_message`），所以前两个词都算动词位。
_VERB_ZONE = 2

_TOKEN_SPLIT = re.compile(r"[^0-9a-zA-Z]+")
_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
# 描述第一句：句号/换行/分号截断，"e.g." 之类不在句首，够用
_FIRST_SENTENCE = re.compile(r"^[^.!?\n;]+")


@dataclass(frozen=True)
class McpToolAccess:
    """一次读写判定的结果，带判据来源（给摘要与日志用）。"""

    is_write: bool
    source: str
    detail: str = ""

    def reason(self) -> str:
        """@return 人可读的判据，如 `annotation:readOnlyHint=false`"""
        return f"{self.source}:{self.detail}" if self.detail else self.source


def split_tool_name(name: str) -> list[str]:
    """
    把工具名切成小写词：下划线 / 连字符 / 点 / 驼峰都算边界。

    @param name MCP 工具名，如 `git_commit`、`createOrUpdateFile`
    @return 小写词列表
    """
    parts: list[str] = []
    for chunk in _TOKEN_SPLIT.split(name or ""):
        if not chunk:
            continue
        parts.extend(p.lower() for p in _CAMEL_SPLIT.split(chunk) if p)
    return parts


def _word_stems(word: str) -> tuple[str, ...]:
    """
    给一个英文词生成候选词干（只服务于描述起始动词匹配）。

    故意不对**工具名**做词干还原：`git_diff_unstaged` 的 `unstaged`
    一旦被还原成 `unstage`，一个只读的 diff 工具就会被判成写。

    @param word 小写单词
    @return 候选词干（含原词）
    """
    out = [word]
    if word.endswith("ies") and len(word) > 4:
        out.append(word[:-3] + "y")
    if word.endswith("es") and len(word) > 3:
        out.append(word[:-2])
    if word.endswith("s") and len(word) > 2:
        out.append(word[:-1])
    if word.endswith("ed") and len(word) > 3:
        out.append(word[:-2])
        out.append(word[:-1])
    if word.endswith("ing") and len(word) > 4:
        out.append(word[:-3])
        out.append(word[:-3] + "e")
    return tuple(out)


def _match_override(name: str, patterns: object) -> str | None:
    """@return 命中的 glob；未命中或配置不是字符串序列时为 None"""
    if not patterns or isinstance(patterns, (str, bytes)):
        return None
    try:
        items = list(patterns)
    except TypeError:
        return None
    lowered = (name or "").lower()
    for item in items:
        if not isinstance(item, str) or not item.strip():
            continue
        pattern = item.strip()
        if fnmatchcase(lowered, pattern.lower()):
            return pattern
    return None


def _annotation_flag(annotations: object, *names: str) -> bool | None:
    """从标注里取布尔字段，兼容 dict 与 pydantic 模型、camel 与 snake 两套命名。"""
    if annotations is None:
        return None
    for key in names:
        value: object = None
        if isinstance(annotations, dict):
            value = annotations.get(key)
        else:
            value = getattr(annotations, key, None)
        if isinstance(value, bool):
            return value
    return None


def _classify_by_name(name: str) -> McpToolAccess | None:
    tokens = split_tool_name(name)
    if not tokens:
        return None
    zone = tokens[:_VERB_ZONE]
    # 动词位里**先出现**的那个词说话：`get_commit` / `get_post` 的第二个词是名词，
    # 不是动作；反过来 `git_commit` / `slack_post_message` 的第二个词才是动作。
    for token in zone:
        if token in _HARD_WRITE_VERBS:
            return McpToolAccess(True, SOURCE_NAME, token)
        if token in _READ_VERBS:
            return McpToolAccess(False, SOURCE_NAME, token)
    for token in zone:
        if token in _SOFT_WRITE_VERBS:
            return McpToolAccess(True, SOURCE_NAME, token)
    for token in tokens[_VERB_ZONE:]:
        if token in _HARD_WRITE_VERBS:
            return McpToolAccess(True, SOURCE_NAME, token)
    for token in tokens[_VERB_ZONE:]:
        if token in _READ_VERBS:
            return McpToolAccess(False, SOURCE_NAME, token)
    return None


def _classify_by_description(description: str) -> McpToolAccess | None:
    """只看第一句的起始动词。整段扫关键词的误判面太大，不做。"""
    sentence = _FIRST_SENTENCE.match((description or "").strip())
    if sentence is None:
        return None
    words = [w for w in _TOKEN_SPLIT.split(sentence.group(0)) if w]
    for raw in words[:3]:
        word = raw.lower()
        # 跳过 "Recursively search ..." 这类前置副词
        if word.endswith("ly") and word not in _DESCRIPTION_WRITE_VERBS:
            continue
        for stem in _word_stems(word):
            if stem in _DESCRIPTION_WRITE_VERBS:
                return McpToolAccess(True, SOURCE_DESCRIPTION, stem)
            if stem in _READ_VERBS:
                return McpToolAccess(False, SOURCE_DESCRIPTION, stem)
        break
    return None


def classify_mcp_tool(
    name: str,
    description: str = "",
    *,
    annotations: object = None,
    read_tools: object = (),
    write_tools: object = (),
) -> McpToolAccess:
    """
    判断一个 MCP 工具是读还是写，并给出判据来源。

    优先级：用户覆盖 → 服务端标注 → 工具名动词 → 描述起始动词 → 默认按读。

    默认按读是刻意的：判不出来的工具多半真的是读（`fetch`、`sequentialthinking`、
    `mysql_query`），默认按写会把只读会话的外部能力大面积摘掉。
    漏放的那部分靠上面三层收敛，真拦不住的交给用户覆盖。

    @param name MCP 工具名
    @param description MCP 工具描述
    @param annotations MCP `Tool.annotations`（dict 或 pydantic 模型）
    @param read_tools 强制按读处理的工具名 glob 列表
    @param write_tools 强制按写处理的工具名 glob 列表
    @return 判定结果
    """
    hit = _match_override(name, write_tools)
    if hit is not None:
        return McpToolAccess(True, SOURCE_OVERRIDE, hit)
    hit = _match_override(name, read_tools)
    if hit is not None:
        return McpToolAccess(False, SOURCE_OVERRIDE, hit)

    read_only = _annotation_flag(annotations, "readOnlyHint", "read_only_hint")
    if read_only is not None:
        return McpToolAccess(
            not read_only, SOURCE_ANNOTATION, f"readOnlyHint={str(read_only).lower()}"
        )
    destructive = _annotation_flag(annotations, "destructiveHint", "destructive_hint")
    if destructive:
        return McpToolAccess(True, SOURCE_ANNOTATION, "destructiveHint=true")

    by_name = _classify_by_name(name)
    if by_name is not None:
        return by_name
    by_desc = _classify_by_description(description)
    if by_desc is not None:
        return by_desc
    return McpToolAccess(False, SOURCE_DEFAULT)

