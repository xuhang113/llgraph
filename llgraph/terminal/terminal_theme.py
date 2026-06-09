"""经典终端输出主题：纯文本报告 → ANSI 分色。"""

from __future__ import annotations

import re

from llgraph.terminal.style import sty

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_CMD_LINE_RE = re.compile(r"^  (\S(?:.*?\S)?)\s{2,}(.+)$")
_KV_LINE_RE = re.compile(r"^  ([^:]+):\s*(.+)$")
_KV_PLAIN_RE = re.compile(r"^([^:\n]{1,48}):\s+(.+)$")
_CMD_ONLY_RE = re.compile(r"^(?:cd\s+\S+|llgraph(?:\s|$)|>\s)")
_DESC_START_RE = re.compile(
    r"[\u4e00-\u9fff「(【同/↑>]|"
    r"[A-Za-z][\w.-]*(?:\s|[\u4e00-\u9fff「(])"
)
_GROUP_RE = re.compile(r"^  \[(.+)\]$")
_TREE_RE = re.compile(r"^(\s*)([├└]─)(.*)$")
_SESSION_CURRENT_RE = re.compile(r"^(● )(.+)$")

_HINT_PREFIXES = (
    "说明:",
    "说明：",
    "恢复:",
    "恢复：",
    "会话内:",
    "会话内：",
    "新建:",
    "改标题:",
    "删除:",
    "更多:",
    "下次恢复:",
    "落盘目录:",
    "执行日志:",
    "当前模型:",
    "最近一轮:",
    "命令:",
    "日志:",
    "可选:",
    "提示:",
    "切换:",
    "凭据:",
    "配置:",
    "合并规则:",
    "运行时覆盖:",
    "仅用户级",
)
_MILESTONE_PREFIXES = ("▶ ", "▼ ", "✓ ")


def _has_ansi(text: str) -> bool:
    return bool(_ANSI_RE.search(text))


def _infer_value_style(val: str) -> str:
    """
    按文案语义推断 value 样式。

    @param val 展示值
    @return STYLES 键名
    """
    if any(x in val for x in ("失败", "错误", "无法", "不是目录")):
        return "err"
    if any(
        x in val
        for x in (
            "未开启",
            "未启用",
            "未加载",
            "未运行",
            "只读",
            "（无）",
            "(无)",
            "未展示",
        )
    ):
        return "warn"
    if any(
        x in val
        for x in ("已开启", "已启用", "已启动", "可写", "✓", "已在运行", "已切换")
    ):
        return "ok"
    return "value"


def _colorize_kv(key: str, val: str, *, indent: str = "  ") -> str:
    """
    着色 key: value 行。

    @param key 键名
    @param val 值
    @param indent 行首缩进
    @return 着色行
    """
    return f"{indent}{sty(key + ':', 'label')} {sty(val.strip(), _infer_value_style(val))}"


def _format_cmd_desc_line(cmd: str, desc: str) -> str:
    """
    着色「命令 + 说明」行（与 /help 列对齐）。

    @param cmd 命令片段
    @param desc 说明
    @return 着色行
    """
    cmd = cmd.strip()
    desc = desc.strip()
    pad = max(1, 22 - len(cmd))
    return f"  {sty(cmd, 'cmd')}{' ' * pad}{sty(desc, 'hint')}"


def _looks_like_desc(text: str) -> bool:
    """
    判断片段是否像命令说明（而非命令参数续行）。

    @param text 候选说明
    @return 是否像说明
    """
    text = text.strip()
    if len(text) < 2:
        return False
    return bool(_DESC_START_RE.match(text))


def _split_cmd_desc_single_space(body: str) -> tuple[str, str] | None:
    """
    在说明前仅单空格时，从右向左拆分命令与说明。

    @param body 去掉行首两空格后的正文
    @return (cmd, desc) 或 None
    """
    for index in range(len(body) - 1, 0, -1):
        if body[index] != " ":
            continue
        cmd = body[:index].strip()
        desc = body[index + 1 :].strip()
        if not cmd or not desc:
            continue
        if _looks_like_desc(desc):
            return cmd, desc
    return None


def _match_cmd_desc_line(line: str) -> tuple[str, str] | None:
    """
    解析两空格缩进的「命令  说明」行（支持说明前仅单空格）。

    @param line 一行文本
    @return (cmd, desc) 或 None
    """
    if not line.startswith("  ") or line.startswith("    "):
        return None

    matched = _CMD_LINE_RE.match(line)
    if matched:
        return matched.group(1).strip(), matched.group(2).strip()

    body = line[2:]
    gap = re.match(r"^(.+?\S)\s{2,}(.+)$", body)
    if gap:
        return gap.group(1).strip(), gap.group(2).strip()

    return _split_cmd_desc_single_space(body)


def _colorize_terminal_line(line: str) -> str:
    """
    为单行终端输出分色。

    @param line 纯文本一行
    @return 带 ANSI 的一行（无着色时原样）
    """
    if not line.strip():
        return line
    if _has_ansi(line):
        return line

    stripped = line.strip()
    raw = line

    if stripped.startswith("=") or (
        len(stripped) >= 3 and set(stripped) <= {"=", "-"}
    ):
        return sty(stripped, "dim")

    if stripped.startswith("【"):
        return sty(stripped, "accent")

    if stripped.startswith("▸ "):
        return sty(stripped, "brand")

    if stripped.startswith(_MILESTONE_PREFIXES):
        return sty(stripped, "accent")

    if stripped.startswith(">"):
        return sty(stripped, "prompt")

    if stripped.startswith("---") and len(stripped) <= 24:
        return sty(stripped, "dim")

    if stripped.startswith("● 错误") or stripped.startswith("错误:"):
        return sty(stripped, "err")

    tree = _TREE_RE.match(line)
    if tree:
        indent, branch, rest = tree.group(1), tree.group(2), tree.group(3)
        if ":" in rest:
            key, _, val = rest.strip().partition(":")
            key = key.strip()
            val = val.strip()
            label_part = sty(f"{key:<14}", "label") if key else ""
            val_part = sty(val, _infer_value_style(val)) if val else ""
            mid = f" {label_part} {val_part}".rstrip() if key else rest
            return indent + sty(branch, "hint") + mid
        return indent + sty(branch, "hint") + rest

    session = _SESSION_CURRENT_RE.match(stripped)
    if session:
        return sty(session.group(1), "accent") + sty(session.group(2), "value")

    group = _GROUP_RE.match(line)
    if group:
        return f"  {sty('[' + group.group(1) + ']', 'accent')}"

    if line.startswith("      ") and stripped:
        return "      " + sty(stripped, "hint")

    if line.startswith("    ") and not line.startswith("      ") and stripped:
        if "  (" in stripped:
            name, _, rest = stripped.partition("  (")
            return f"    {sty(name, 'cmd')}  ({sty(rest.rstrip(')'), 'path')})"
        return f"    {sty(stripped, 'cmd')}"

    kv = _KV_LINE_RE.match(line)
    if kv and not stripped.startswith("/"):
        return _colorize_kv(kv.group(1).strip(), kv.group(2))

    kv_plain = _KV_PLAIN_RE.match(stripped)
    if kv_plain:
        return _colorize_kv(kv_plain.group(1).strip(), kv_plain.group(2), indent="")

    matched_cmd = _match_cmd_desc_line(line)
    if matched_cmd:
        return _format_cmd_desc_line(*matched_cmd)

    for prefix in _HINT_PREFIXES:
        if stripped.startswith(prefix):
            return sty(stripped, "hint")

    if stripped.startswith("（") or stripped.startswith("("):
        return sty(stripped, "hint")

    if any(x in stripped for x in ("运行失败", "无法解析", "未知命令")):
        return sty(stripped, "err")

    if any(
        x in stripped
        for x in (
            "已切换",
            "已启动",
            "已停止",
            "已开启",
            "Survey 交互已",
            "index watch 已在运行",
        )
    ):
        return sty(stripped, "ok")

    if not line.startswith(" ") and not line.startswith("\t"):
        if len(stripped) < 48 and not stripped.endswith("。"):
            return sty(stripped, "title")
        if stripped.endswith("。") or stripped.endswith("）"):
            return sty(stripped, "hint")

    if line.startswith("  ") and not line.startswith("    "):
        if _CMD_ONLY_RE.match(stripped):
            return f"  {sty(stripped, 'cmd')}"
        return f"  {sty(stripped, 'hint')}"

    return line


def colorize_terminal_text(text: str) -> str:
    """
    为终端多行报告分色（/help、/tools、/sessions 等）。

    @param text 纯文本
    @return 带 ANSI 的多行文本
    """
    return "\n".join(_colorize_terminal_line(line) for line in text.splitlines())
