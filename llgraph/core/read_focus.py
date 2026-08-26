"""大文件智能读：大纲 + 本问检索命中窗（对齐 Cursor 大纲 / Claude Code 默认截断）。

无行段地把 2000+ 行带行号全文塞进下一轮 prompt，是改码场景最大的 token 浪费之一：
首 token 变慢、上下文被淹没、模型还以为已经「读过」中间实现。
本模块：小文件或明确行段仍返回编号正文；未指定行段的大文件改为

- 文件头（import / package）
- 符号大纲（class/def/func 行号）
- 本问 grep / parallel 命中窗

模型再对目标符号带 start_line/end_line 精读完整函数/类。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

FOCUS_READ_MARKER = "【read 已折叠】"

# 不超过该行数：无行段时仍返回全文（小模块不值得折叠）
FULL_READ_MAX_LINES = 350
# 折叠读时保留的文件头行数（package / import）
HEAD_LINES = 40
# 命中窗半径
HIT_RADIUS = 35
# 命中窗最多段数 / 合计行数
MAX_HIT_WINDOWS = 6
MAX_HIT_WINDOW_LINES = 180
# 大纲最多条
MAX_OUTLINE_ENTRIES = 80
# 折叠结果字符预算（超出先丢命中窗再截大纲）
MAX_FOCUS_CHARS = 16_000
# 大纲/行标展示时单行截断
_OUTLINE_LINE_MAX = 100

_PY_SYMBOL = re.compile(
    r"^([ \t]*)((?:async[ \t]+)?def[ \t]+\w+.*|class[ \t]+\w+.*)$"
)
_JAVA_TYPE = re.compile(
    r"^([ \t]*)((?:(?:public|protected|private|abstract|static|final|sealed"
    r"|non-sealed)[ \t]+)*(?:class|interface|enum|record)[ \t]+\w+.*)$"
)
_JAVA_METHOD = re.compile(
    r"^([ \t]*)(?:(?:public|protected|private|static|final|synchronized|"
    r"native|default|abstract)[ \t]+)+[\w.<>,\[\]?]+\s+\w+\s*\(.*$"
)
_JS_SYMBOL = re.compile(
    r"^([ \t]*)((?:export[ \t]+)?(?:default[ \t]+)?(?:async[ \t]+)?(?:function\*?"
    r"|class|interface|type|enum|namespace)[ \t]+\w+.*"
    r"|(?:export[ \t]+)?(?:const|let|var)[ \t]+\w+[ \t]*=[ \t]*(?:async[ \t]+)?"
    r"(?:\(|function).*)$"
)
_GO_SYMBOL = re.compile(
    r"^([ \t]*)((?:func[ \t]+.*)|(?:type[ \t]+\w+.*))$"
)
_RUST_SYMBOL = re.compile(
    r"^([ \t]*)((?:pub(?:\([^)]+\))?[ \t]+)?(?:async[ \t]+)?(?:unsafe[ \t]+)?"
    r"fn[ \t]+\w+.*|(?:pub(?:\([^)]+\))?[ \t]+)?(?:struct|enum|trait|mod|impl|type)"
    r"[ \t]+.*)$"
)
_MD_HEADING = re.compile(r"^(#{1,3})[ \t]+\S.*$")
_GENERIC_SKIP_PREFIX = (
    "#",
    "//",
    "/*",
    "*",
    "}",
    ")",
    "]",
    "import ",
    "from ",
    "package ",
    "using ",
    "use ",
)


@dataclass(frozen=True)
class OutlineEntry:
    """大纲中的一个符号。"""

    line: int
    text: str
    indent: int = 0


def should_focus_read(
    *,
    start_line: int,
    end_line: int,
    total_lines: int,
) -> bool:
    """
    是否把这次 read 折叠为大纲+命中窗。

    仅「从文件开头读到末尾」且超过 FULL_READ_MAX_LINES 时折叠。
    明确行段或续读（start_line>1）仍返回编号正文。

    @param start_line 起始行（已 clamp 到 >=1）
    @param end_line 结束行；0 表示到末尾
    @param total_lines 文件总行数
    @return 是否折叠
    """
    if total_lines <= FULL_READ_MAX_LINES:
        return False
    if start_line > 1:
        return False
    if end_line > 0 and end_line < total_lines:
        return False
    return True


def format_numbered_slice(
    path: str,
    lines: list[str],
    start: int,
    end: int,
    *,
    total: int | None = None,
) -> str:
    """
    格式化为带行号正文（与历史 read_file 头格式兼容）。

    @param path 展示用相对路径
    @param lines 文件全部行
    @param start 起始行号（1-based，含）
    @param end 结束行号（1-based，含）
    @param total 总行数；默认 len(lines)
    @return 文本
    """
    total_n = total if total is not None else len(lines)
    start_n = max(1, start)
    end_n = min(end, len(lines))
    if start_n > end_n or not lines:
        return f"--- {path} (行 {start_n}-{end_n} / 共 {total_n} 行) ---\n"
    header = f"--- {path} (行 {start_n}-{end_n} / 共 {total_n} 行) ---\n"
    body = "\n".join(
        f"{start_n + i}| {line}" for i, line in enumerate(lines[start_n - 1 : end_n])
    )
    return header + body


def extract_outline(lines: list[str], path: str) -> list[OutlineEntry]:
    """
    按后缀提取符号行；无匹配时退回列 0 路标。

    @param lines 源文件行
    @param path 相对路径（看后缀）
    @return 大纲条目（已按上限截断）
    """
    suffix = Path(path or "").suffix.lower()
    if suffix in {".py", ".pyi", ".pyw"}:
        entries = _scan_regex(lines, _PY_SYMBOL)
    elif suffix in {".java", ".kt", ".kts"}:
        entries = _scan_java(lines)
    elif suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        entries = _scan_regex(lines, _JS_SYMBOL)
    elif suffix == ".go":
        entries = _scan_regex(lines, _GO_SYMBOL)
    elif suffix == ".rs":
        entries = _scan_regex(lines, _RUST_SYMBOL)
    elif suffix in {".md", ".mdc"}:
        entries = _scan_markdown(lines)
    else:
        entries = _scan_generic(lines)
    if not entries:
        entries = _scan_generic(lines)
    return _cap_outline(entries)


def hit_lines_for_path(rel_path: str) -> list[int]:
    """
    从当前工具执行上下文收集本问对该路径的检索命中行。

    @param rel_path 正在读取的相对路径
    @return 升序去重行号
    """
    from llgraph.context.search_hit_lines import collect_search_hits_from_messages
    from llgraph.core.tool_execution_context import get_tool_execution_messages

    messages = get_tool_execution_messages()
    if not messages:
        return []
    hits = collect_search_hits_from_messages(messages)
    if not hits:
        return []
    want = _norm_rel(rel_path)
    found: set[int] = set()
    for raw_path, line_nos in hits.items():
        if _paths_match(want, _norm_rel(raw_path)):
            found.update(line_nos)
    return sorted(found)


def format_focus_read(
    path: str,
    lines: list[str],
    *,
    hit_lines: list[int] | None = None,
) -> str:
    """
    渲染折叠读结果。

    @param path 相对路径
    @param lines 源文件全部行
    @param hit_lines 本问检索命中行号
    @return 给模型的工具正文
    """
    total = len(lines)
    chars = sum(len(line) + 1 for line in lines)
    outline = extract_outline(lines, path)
    windows = plan_hit_windows(hit_lines or [], total_lines=total)
    head_end = min(HEAD_LINES, total)

    parts: list[str] = [
        (
            f"{FOCUS_READ_MARKER}`{path}` 共 {total} 行 / {chars} 字符；"
            "未把全文注入上下文（对齐 Cursor 大纲 / Claude Code 默认截断）。"
        ),
        (
            "精读某符号请 read_file(path, start_line, end_line) "
            "一次取完整函数或类（约 80–300 行）；禁止再无行段整文件 read。"
        ),
        "",
        format_numbered_slice(path, lines, 1, head_end, total=total),
    ]

    if outline:
        parts.append("")
        parts.append(f"## 符号大纲（{len(outline)} 个）")
        parts.extend(_format_outline_lines(outline))

    for start, end in windows:
        if _range_inside(start, end, 1, head_end):
            continue
        parts.append("")
        parts.append(format_numbered_slice(path, lines, start, end, total=total))

    if not outline and not windows:
        landmarks = _line_landmarks(lines, skip_until=head_end)
        if landmarks:
            parts.append("")
            parts.append("## 行标")
            parts.extend(_format_outline_lines(landmarks))

    text = "\n".join(parts).rstrip() + "\n"
    if len(text) <= MAX_FOCUS_CHARS:
        return text
    return _shrink_focus_text(
        path,
        lines,
        outline=outline,
        windows=windows,
        head_end=head_end,
        total=total,
        chars=chars,
    )


def plan_hit_windows(
    hit_lines: list[int],
    *,
    total_lines: int,
    radius: int = HIT_RADIUS,
    max_windows: int = MAX_HIT_WINDOWS,
    max_lines: int = MAX_HIT_WINDOW_LINES,
) -> list[tuple[int, int]]:
    """
    把命中行扩成合并后的行窗，并限制段数与总行数。

    @param hit_lines 命中行号
    @param total_lines 文件总行数
    @param radius 上下半径
    @param max_windows 最多窗数
    @param max_lines 合计行数上限
    @return [(start,end), ...] 含端、已合并
    """
    if not hit_lines or total_lines <= 0 or radius < 0:
        return []
    raw: list[tuple[int, int]] = []
    for line_no in hit_lines:
        if line_no < 1 or line_no > total_lines:
            continue
        start = max(1, line_no - radius)
        end = min(total_lines, line_no + radius)
        raw.append((start, end))
    merged = _merge_ranges(raw)
    picked: list[tuple[int, int]] = []
    used = 0
    for start, end in merged:
        if len(picked) >= max_windows:
            break
        span = end - start + 1
        if used + span > max_lines:
            remain = max_lines - used
            if remain < 12:
                break
            end = start + remain - 1
            span = remain
        picked.append((start, end))
        used += span
    return picked


def _scan_regex(lines: list[str], pattern: re.Pattern[str]) -> list[OutlineEntry]:
    entries: list[OutlineEntry] = []
    for idx, line in enumerate(lines, start=1):
        match = pattern.match(line.rstrip())
        if not match:
            continue
        indent = len(match.group(1) or "")
        text = (match.group(2) if match.lastindex and match.lastindex >= 2 else line).strip()
        entries.append(OutlineEntry(line=idx, text=text[:_OUTLINE_LINE_MAX], indent=indent))
    return entries


def _scan_java(lines: list[str]) -> list[OutlineEntry]:
    entries: list[OutlineEntry] = []
    for idx, line in enumerate(lines, start=1):
        stripped = line.rstrip()
        match = _JAVA_TYPE.match(stripped) or _JAVA_METHOD.match(stripped)
        if not match:
            continue
        indent = len(match.group(1) or "")
        text = stripped.strip()[:_OUTLINE_LINE_MAX]
        entries.append(OutlineEntry(line=idx, text=text, indent=indent))
    return entries


def _scan_markdown(lines: list[str]) -> list[OutlineEntry]:
    entries: list[OutlineEntry] = []
    for idx, line in enumerate(lines, start=1):
        match = _MD_HEADING.match(line.rstrip())
        if not match:
            continue
        hashes = match.group(1) or "#"
        indent = max(0, len(hashes) - 1) * 2
        entries.append(
            OutlineEntry(line=idx, text=line.strip()[:_OUTLINE_LINE_MAX], indent=indent)
        )
    return entries


def _scan_generic(lines: list[str]) -> list[OutlineEntry]:
    entries: list[OutlineEntry] = []
    for idx, line in enumerate(lines, start=1):
        if line[:1] in {" ", "\t"}:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(_GENERIC_SKIP_PREFIX):
            continue
        if stripped[0] in {"{", "(", "[", "<"}:
            continue
        entries.append(OutlineEntry(line=idx, text=stripped[:_OUTLINE_LINE_MAX], indent=0))
        if len(entries) >= MAX_OUTLINE_ENTRIES * 2:
            break
    return entries


def _cap_outline(
    entries: list[OutlineEntry],
    limit: int = MAX_OUTLINE_ENTRIES,
) -> list[OutlineEntry]:
    if len(entries) <= limit:
        return entries
    ranked = sorted(entries, key=lambda item: (item.indent, item.line))
    keep_lines = {item.line for item in ranked[:limit]}
    return [item for item in entries if item.line in keep_lines]


def _format_outline_lines(entries: list[OutlineEntry]) -> list[str]:
    return [f"  {item.line}| {item.text}" for item in entries]


def _line_landmarks(lines: list[str], *, skip_until: int, step: int = 50) -> list[OutlineEntry]:
    """无符号时每隔 step 行给一个路标。"""
    entries: list[OutlineEntry] = []
    total = len(lines)
    line_no = skip_until + step
    while line_no <= total:
        text = lines[line_no - 1].strip() or "(空行)"
        entries.append(OutlineEntry(line=line_no, text=text[:_OUTLINE_LINE_MAX], indent=0))
        line_no += step
    return entries[:24]


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda item: (item[0], item[1]))
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _range_inside(start: int, end: int, outer_start: int, outer_end: int) -> bool:
    return start >= outer_start and end <= outer_end


def _norm_rel(path: str) -> str:
    text = (path or "").replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text.strip("/")


def _paths_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    return a.endswith("/" + b) or b.endswith("/" + a)


def _shrink_focus_text(
    path: str,
    lines: list[str],
    *,
    outline: list[OutlineEntry],
    windows: list[tuple[int, int]],
    head_end: int,
    total: int,
    chars: int,
) -> str:
    """超预算时丢掉命中窗，只留头+浅大纲。"""
    shallow = [item for item in outline if item.indent == 0][:40] or outline[:40]
    parts = [
        (
            f"{FOCUS_READ_MARKER}`{path}` 共 {total} 行 / {chars} 字符；"
            "未把全文注入上下文（对齐 Cursor 大纲 / Claude Code 默认截断）。"
        ),
        "精读某符号请 read_file(path, start_line, end_line) 一次取完整函数或类。",
        "",
        format_numbered_slice(path, lines, 1, head_end, total=total),
    ]
    if shallow:
        parts.append("")
        parts.append(f"## 符号大纲（{len(shallow)} 个，已压缩）")
        parts.extend(_format_outline_lines(shallow))
    if windows:
        parts.append("")
        parts.append(
            f"（另有 {len(windows)} 段检索命中窗因长度省略；请按大纲行号精读。）"
        )
    return "\n".join(parts).rstrip() + "\n"
