"""search_replace 容错匹配：对齐 Cursor / Claude Code / Codex 的改码命中路径。

优先 exact；失败则依次尝试换行符、读文件行号前缀、行尾空白、缩进对齐、行内空白折叠。
仍 0 命中时，若全文件有**唯一**高相似行窗（old_string 笔误、中间多/少一行），
按该窗落地——对标 Cursor Fast Apply / Codex apply_patch 的 fuzzy 命中，避免再空转一轮 LLM。
同一文件多个 hunk 按顺序作用在内存文本上，全部成功后再落盘。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

_READ_LINE_PREFIX = re.compile(r"^\d+\| ")
_READ_HEADER = re.compile(
    r"^---\s+.+\s+\(行\s+\d+-\d+\s+/ 共\s+\d+\s+行\)(?:\s+\[[^\]]+\])?(?:\s+---)?\s*\n?",
)
_WS_RUN = re.compile(r"[^\S\n]+")

_STRATEGY_ORDER = (
    "exact",
    "newline",
    "trailing_ws",
    "indent",
    "whitespace",
)

# unique fuzzy：仅在精确族全部失败后启用；replace_all 不用（误伤面太大）
_FUZZY_MAX_NEEDLE_LINES = 80
_FUZZY_FULL_SCAN_LINES = 8000
_FUZZY_MIN_GAP = 0.12
_FUZZY_MIN_FIRST_LINE_RATIO = 0.70
_FUZZY_MIN_SINGLE_LINE_CHARS = 12


@dataclass(frozen=True)
class EditHunk:
    """单次替换片段。"""

    old_string: str
    new_string: str
    replace_all: bool = False


@dataclass(frozen=True)
class EditHit:
    """一次命中在原文中的字符区间。"""

    start: int
    end: int
    matched: str
    strategy: str
    indent_kind: str = ""
    indent_ws: str = ""
    ratio: float = 0.0


@dataclass
class ApplyEditResult:
    """对整段文本应用一个或多个 hunk 的结果。"""

    ok: bool
    new_text: str = ""
    replacements: int = 0
    hunks_applied: int = 0
    strategy: str = "exact"
    error: str = ""
    hint: str = ""
    tried: tuple[str, ...] = field(default_factory=tuple)
    fuzzy_ratio: float = 0.0


def apply_search_replace(
    text: str,
    old_string: str,
    new_string: str,
    *,
    replace_all: bool = False,
    require_unique: bool = True,
    allow_fuzzy: bool = True,
) -> ApplyEditResult:
    """
    对单段文本做一次容错替换。

    @param text 当前文件内容
    @param old_string 待匹配片段
    @param new_string 替换文本
    @param replace_all 是否替换全部命中
    @param require_unique 非 replace_all 时是否要求唯一命中
    @param allow_fuzzy 精确族失败后是否尝试 unique fuzzy 窗口
    @return 应用结果（失败时不改原文）
    """
    return apply_edit_hunks(
        text,
        [EditHunk(old_string, new_string, replace_all=replace_all)],
        require_unique=require_unique,
        allow_fuzzy=allow_fuzzy,
    )


def apply_edit_hunks(
    text: str,
    hunks: list[EditHunk],
    *,
    require_unique: bool = True,
    allow_fuzzy: bool = True,
) -> ApplyEditResult:
    """
    按顺序应用多个 hunk；任一失败则整体失败。

    @param text 当前文件内容
    @param hunks 替换列表
    @param require_unique 非 replace_all 时是否要求唯一命中
    @param allow_fuzzy 精确族失败后是否尝试 unique fuzzy 窗口
    @return 应用结果
    """
    cleaned = [h for h in hunks if (h.old_string or "").strip() or h.old_string == ""]
    if not cleaned:
        return ApplyEditResult(ok=False, error="必须提供 old_string 或 replacements")
    current = text
    total = 0
    strategies: list[str] = []
    all_tried: list[str] = []
    fuzzy_ratio = 0.0
    for idx, hunk in enumerate(cleaned, start=1):
        one = _apply_one_hunk(
            current,
            hunk,
            require_unique=require_unique,
            allow_fuzzy=allow_fuzzy,
        )
        all_tried.extend(one.tried)
        if not one.ok:
            prefix = f"hunk {idx}/{len(cleaned)}: " if len(cleaned) > 1 else ""
            return ApplyEditResult(
                ok=False,
                error=prefix + one.error,
                hint=one.hint,
                tried=tuple(dict.fromkeys(all_tried)),
            )
        current = one.new_text
        total += one.replacements
        strategies.append(one.strategy)
        if one.fuzzy_ratio > fuzzy_ratio:
            fuzzy_ratio = one.fuzzy_ratio
    strategy = strategies[0] if len(set(strategies)) == 1 else "+".join(strategies)
    return ApplyEditResult(
        ok=True,
        new_text=current,
        replacements=total,
        hunks_applied=len(cleaned),
        strategy=strategy,
        tried=tuple(dict.fromkeys(all_tried)),
        fuzzy_ratio=fuzzy_ratio,
    )


def format_apply_failure(rel: str, result: ApplyEditResult) -> str:
    """
    将失败结果格式化为工具返回文案（保留 WriteFailureTracker 可识别标记）。

    @param rel 相对路径
    @param result 失败结果
    @return 工具错误文本
    """
    tried = " / ".join(result.tried) if result.tried else "exact"
    if "不唯一" in result.error:
        return (
            f"old_string 在 {rel} 中出现多次，不唯一。"
            f"{result.error}。请扩大上下文使片段唯一，或设置 replace_all=true。"
        )
    loc = result.error if rel in result.error else f"{result.error}: {rel}"
    lines = [loc]
    if "未找到 old_string" in result.error:
        lines.append(f"已尝试匹配: {tried}")
    if result.hint:
        lines.append(result.hint)
    lines.append(
        "请先 read_file 核对缩进与换行；同一文件多处修改可用 replacements 一次提交。"
    )
    return "\n".join(lines)


_SNIPPET_CONTEXT_LINES = 8
_SNIPPET_MAX_LINES = 48
_SNIPPET_MAX_CHARS = 3500
_SNIPPET_HEAD_LINES = 40


def _changed_span(old_text: str, new_text: str) -> tuple[int, int] | None:
    """
    用公共前后缀定位 new_text 中的改动行区间（0-based, end 不含）。

    @param old_text 写入前
    @param new_text 写入后
    @return (start, end)；无改动则为 None
    """
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    if old_lines == new_lines:
        return None
    prefix = 0
    limit = min(len(old_lines), len(new_lines))
    while prefix < limit and old_lines[prefix] == new_lines[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < (len(old_lines) - prefix)
        and suffix < (len(new_lines) - prefix)
        and old_lines[len(old_lines) - 1 - suffix] == new_lines[len(new_lines) - 1 - suffix]
    ):
        suffix += 1
    start = prefix
    end = len(new_lines) - suffix
    if end < start:
        end = start
    if start >= len(new_lines):
        if not new_lines:
            return None
        start = max(0, len(new_lines) - 1)
        end = len(new_lines)
    return (start, max(end, start + 1))


def changed_line_span(old_text: str, new_text: str) -> tuple[int, int] | None:
    """
    改动行区间（1-based，两端包含），供写入后诊断定位。

    @param old_text 写入前
    @param new_text 写入后
    @return (start, end)；无改动则为 None
    """
    span = _changed_span(old_text, new_text)
    if span is None:
        return None
    start, end = span
    return (start + 1, max(start + 1, end))


def format_write_snapshot(
    rel: str,
    text: str,
    *,
    start_line: int = 1,
    end_line: int = 0,
    label: str = "写入后快照",
) -> str:
    """
    把文件片段格式化成带行号的快照（与 read_file 同类，便于模型接着改）。

    @param rel 相对路径
    @param text 当前文件全文
    @param start_line 起始行（1-based）
    @param end_line 结束行（含）；0 表示到末尾
    @param label 快照标记
    @return 快照正文；空文件给一行说明
    """
    lines = text.splitlines()
    total = len(lines)
    if total == 0:
        return f"--- {rel} (空文件，0 行) [{label}] ---"
    start = max(1, start_line)
    end = total if end_line <= 0 else min(end_line, total)
    if start > total:
        start = 1
        end = min(total, _SNIPPET_HEAD_LINES)
    selected = lines[start - 1 : end]
    if len(selected) > _SNIPPET_MAX_LINES:
        head_n = _SNIPPET_MAX_LINES // 2
        tail_n = _SNIPPET_MAX_LINES - head_n
        head = selected[:head_n]
        tail = selected[-tail_n:]
        skip = len(selected) - head_n - tail_n
        numbered: list[str] = [
            f"{start + i}| {line}" for i, line in enumerate(head)
        ]
        numbered.append(f"…（省略 {skip} 行）")
        tail_start = end - tail_n + 1
        numbered.extend(f"{tail_start + i}| {line}" for i, line in enumerate(tail))
        body = "\n".join(numbered)
        shown_end = end
    else:
        body = "\n".join(f"{start + i}| {line}" for i, line in enumerate(selected))
        shown_end = start + len(selected) - 1
    if len(body) > _SNIPPET_MAX_CHARS:
        body = body[: _SNIPPET_MAX_CHARS - 20].rstrip() + "\n…（快照已截断）"
    return (
        f"--- {rel} (行 {start}-{shown_end} / 共 {total} 行) [{label}] ---\n"
        f"{body}\n"
        "后续 search_replace 请以此快照为准，勿使用写入前的 read。"
    )


def format_post_edit_snapshot(rel: str, old_text: str, new_text: str) -> str:
    """
    根据改动区间生成写入后快照。

    @param rel 相对路径
    @param old_text 写入前
    @param new_text 写入后
    @return 快照；无法定位时退回文件头部
    """
    span = _changed_span(old_text, new_text)
    lines = new_text.splitlines()
    total = len(lines)
    if not span or total == 0:
        return format_write_snapshot(
            rel, new_text, start_line=1, end_line=min(total, _SNIPPET_HEAD_LINES) or 0
        )
    start, end = span
    ctx_start = max(0, start - _SNIPPET_CONTEXT_LINES)
    ctx_end = min(total, end + _SNIPPET_CONTEXT_LINES)
    return format_write_snapshot(
        rel, new_text, start_line=ctx_start + 1, end_line=ctx_end
    )


def format_apply_success(
    rel: str,
    result: ApplyEditResult,
    *,
    old_text: str = "",
) -> str:
    """
    成功替换后的工具返回文案（首行前缀供失败追踪识别）。

    @param rel 相对路径
    @param result 成功结果
    @param old_text 写入前全文，用于生成当前片段
    @return 工具成功文本
    """
    hunk_bit = ""
    if result.hunks_applied > 1:
        hunk_bit = f" / {result.hunks_applied} hunk"
    strategy_bit = ""
    if result.strategy == "fuzzy":
        pct = f"{result.fuzzy_ratio:.0%}" if result.fuzzy_ratio else ""
        strategy_bit = f"，匹配策略=fuzzy{f'({pct})' if pct else ''}"
    elif result.strategy and result.strategy != "exact" and "exact" not in result.strategy.split("+"):
        strategy_bit = f"，匹配策略={result.strategy}"
    elif result.strategy and "+" in result.strategy and result.strategy != "exact":
        parts = result.strategy.split("+")
        if any(p != "exact" for p in parts):
            strategy_bit = f"，匹配策略={result.strategy}"
    header = f"已替换 {rel}（{result.replacements} 处{hunk_bit}{strategy_bit}）"
    if not result.new_text and not old_text:
        return header
    snapshot = format_post_edit_snapshot(rel, old_text, result.new_text)
    return f"{header}\n{snapshot}"


def strip_read_file_artifacts(text: str) -> str:
    """
    去掉 read_file 输出的表头与 `行号| ` 前缀（模型常把工具输出整段贴进 old_string）。

    @param text 可能含行号前缀的片段
    @return 清洗后文本；无明显前缀时原样返回
    """
    body = text
    header = _READ_HEADER.match(body)
    if header:
        body = body[header.end() :]
    lines = body.split("\n")
    nonempty = [ln for ln in lines if ln.strip()]
    if not nonempty:
        return body if header else text
    prefixed = sum(1 for ln in nonempty if _READ_LINE_PREFIX.match(ln))
    if prefixed < max(1, int(len(nonempty) * 0.6)):
        return body if header else text
    stripped = [_READ_LINE_PREFIX.sub("", ln, count=1) for ln in lines]
    return "\n".join(stripped)


def nearest_snippet_hint(text: str, needle: str, *, limit: int = 8) -> str:
    """
    old_string 未命中时给出最相近行窗与首行子串命中。

    @param text 文件全文
    @param needle 待匹配片段
    @param limit 最多展示行数
    @return 提示文本；无线索时为空
    """
    cleaned = strip_read_file_artifacts(needle)
    first_line = ""
    for raw in cleaned.splitlines():
        if raw.strip():
            first_line = raw.strip()
            break
    hints: list[str] = []
    if first_line and len(first_line) >= 4:
        for idx, line in enumerate(text.splitlines(), 1):
            if first_line in line or _collapse_ws(first_line) in _collapse_ws(line):
                hints.append(f"  L{idx}: {line[:120]}")
                if len(hints) >= 5:
                    break
    window_hint = _best_window_hint(text, cleaned, limit=limit)
    parts: list[str] = []
    if hints:
        parts.append("相近行:\n" + "\n".join(hints))
    if window_hint:
        parts.append(window_hint)
    return "\n".join(parts)


def _apply_one_hunk(
    text: str,
    hunk: EditHunk,
    *,
    require_unique: bool,
    allow_fuzzy: bool = True,
) -> ApplyEditResult:
    needle = hunk.old_string
    if needle == "":
        return ApplyEditResult(ok=False, error="错误: search_replace 缺少 old_string")
    needles = [needle]
    stripped = strip_read_file_artifacts(needle)
    if stripped != needle:
        needles.append(stripped)

    tried: list[str] = []
    for candidate in needles:
        for strategy in _STRATEGY_ORDER:
            tried.append(strategy)
            hits = _find_hits(text, candidate, strategy)
            if not hits:
                continue
            if not hunk.replace_all and require_unique and len(hits) != 1:
                return ApplyEditResult(
                    ok=False,
                    error=(
                        f"old_string 不唯一（{len(hits)} 处匹配）"
                    ),
                    hint=nearest_snippet_hint(text, candidate),
                    tried=tuple(dict.fromkeys(tried)),
                    strategy=strategy,
                )
            chosen = hits if hunk.replace_all else hits[:1]
            new_text = _replace_hits(text, chosen, hunk.new_string)
            return ApplyEditResult(
                ok=True,
                new_text=new_text,
                replacements=len(chosen),
                hunks_applied=1,
                strategy=strategy,
                tried=tuple(dict.fromkeys(tried)),
            )

    if allow_fuzzy and not hunk.replace_all:
        tried.append("fuzzy")
        for candidate in reversed(needles):
            fuzzy_hit = find_fuzzy_unique_hit(text, candidate)
            if fuzzy_hit is None:
                continue
            new_text = _replace_hits(text, [fuzzy_hit], hunk.new_string)
            return ApplyEditResult(
                ok=True,
                new_text=new_text,
                replacements=1,
                hunks_applied=1,
                strategy="fuzzy",
                tried=tuple(dict.fromkeys(tried)),
                fuzzy_ratio=fuzzy_hit.ratio,
            )

    return ApplyEditResult(
        ok=False,
        error="未找到 old_string（0 处匹配）",
        hint=nearest_snippet_hint(text, needle),
        tried=tuple(dict.fromkeys(tried)),
    )


def _find_hits(text: str, needle: str, strategy: str) -> list[EditHit]:
    if strategy == "exact":
        return _hits_exact(text, needle, "exact")
    if strategy == "newline":
        return _hits_newline(text, needle)
    if strategy == "trailing_ws":
        return _hits_linewise(text, needle, "trailing_ws", _norm_trailing_ws)
    if strategy == "indent":
        return _hits_indent(text, needle)
    if strategy == "whitespace":
        return _hits_linewise(text, needle, "whitespace", _norm_whitespace)
    return []


def _hits_exact(text: str, needle: str, strategy: str) -> list[EditHit]:
    if not needle:
        return []
    hits: list[EditHit] = []
    start = 0
    step = max(1, len(needle))
    while True:
        idx = text.find(needle, start)
        if idx < 0:
            break
        hits.append(
            EditHit(
                start=idx,
                end=idx + len(needle),
                matched=needle,
                strategy=strategy,
            )
        )
        start = idx + step
    return hits


def _hits_newline(text: str, needle: str) -> list[EditHit]:
    variants = _newline_variants(needle)
    seen: set[tuple[int, int]] = set()
    hits: list[EditHit] = []
    for variant in variants:
        if variant == needle:
            continue
        for hit in _hits_exact(text, variant, "newline"):
            key = (hit.start, hit.end)
            if key in seen:
                continue
            seen.add(key)
            hits.append(hit)
    hits.sort(key=lambda h: h.start)
    return hits


def _newline_variants(text: str) -> list[str]:
    lf = text.replace("\r\n", "\n").replace("\r", "\n")
    crlf = lf.replace("\n", "\r\n")
    out: list[str] = []
    for item in (text, lf, crlf):
        if item not in out:
            out.append(item)
    return out


def _file_lines(text: str) -> list[str]:
    if text == "":
        return []
    return text.splitlines(keepends=True)


def _line_offsets(lines: list[str]) -> list[int]:
    offsets = [0]
    acc = 0
    for line in lines:
        acc += len(line)
        offsets.append(acc)
    return offsets


def _rstrip_nl(line: str) -> str:
    if line.endswith("\r\n"):
        return line[:-2]
    if line.endswith("\n") or line.endswith("\r"):
        return line[:-1]
    return line


def _norm_trailing_ws(line: str) -> str:
    return _rstrip_nl(line).rstrip(" \t")


def _collapse_ws(line: str) -> str:
    return _WS_RUN.sub(" ", line).strip()


def _norm_whitespace(line: str) -> str:
    return _collapse_ws(_rstrip_nl(line))


def _hits_linewise(
    text: str,
    needle: str,
    strategy: str,
    normalize,
) -> list[EditHit]:
    needle_lines = _file_lines(needle)
    file_lines = _file_lines(text)
    n = len(needle_lines)
    if n == 0 or n > len(file_lines):
        return []
    needle_norm = [normalize(ln) for ln in needle_lines]
    # 末行无换行时，允许与文件末行（可带换行）对齐
    offsets = _line_offsets(file_lines)
    hits: list[EditHit] = []
    for i in range(0, len(file_lines) - n + 1):
        ok = True
        for j in range(n):
            if normalize(file_lines[i + j]) != needle_norm[j]:
                ok = False
                break
        if not ok:
            continue
        start = offsets[i]
        end = offsets[i + n]
        if not needle.endswith(("\n", "\r")):
            last = file_lines[i + n - 1]
            end -= len(last) - len(_rstrip_nl(last))
        matched = text[start:end]
        hits.append(
            EditHit(start=start, end=end, matched=matched, strategy=strategy)
        )
    return hits


def _leading_ws(line: str) -> str:
    body = _rstrip_nl(line).lstrip(" \t")
    raw = _rstrip_nl(line)
    if not body:
        return raw[: len(raw) - len(raw.lstrip(" \t"))]
    return raw[: len(raw) - len(body)]


def _first_nonempty_index(lines: list[str]) -> int:
    for idx, line in enumerate(lines):
        if _rstrip_nl(line).strip():
            return idx
    return 0


def _hits_indent(text: str, needle: str) -> list[EditHit]:
    needle_lines = _file_lines(needle)
    file_lines = _file_lines(text)
    n = len(needle_lines)
    if n == 0 or n > len(file_lines):
        return []
    n_idx = _first_nonempty_index(needle_lines)
    needle_body = _norm_trailing_ws(needle_lines[n_idx]).lstrip(" \t")
    if not needle_body:
        return []
    needle_lead = _leading_ws(needle_lines[n_idx])
    offsets = _line_offsets(file_lines)
    hits: list[EditHit] = []
    for i in range(0, len(file_lines) - n + 1):
        file_anchor = file_lines[i + n_idx]
        if _norm_trailing_ws(file_anchor).lstrip(" \t") != needle_body:
            continue
        file_lead = _leading_ws(file_anchor)
        kind, ws = _indent_delta(file_lead, needle_lead)
        shifted = [_apply_indent_to_line(ln, kind, ws) for ln in needle_lines]
        ok = True
        for j in range(n):
            if _norm_trailing_ws(file_lines[i + j]) != _norm_trailing_ws(shifted[j]):
                ok = False
                break
        if not ok:
            continue
        start = offsets[i]
        end = offsets[i + n]
        if not needle.endswith(("\n", "\r")):
            last = file_lines[i + n - 1]
            end -= len(last) - len(_rstrip_nl(last))
        hits.append(
            EditHit(
                start=start,
                end=end,
                matched=text[start:end],
                strategy="indent",
                indent_kind=kind,
                indent_ws=ws,
            )
        )
    return hits


def _indent_delta(file_lead: str, needle_lead: str) -> tuple[str, str]:
    if file_lead == needle_lead:
        return ("add", "")
    if file_lead.startswith(needle_lead):
        return ("add", file_lead[len(needle_lead) :])
    if needle_lead.startswith(file_lead):
        return ("remove", needle_lead[len(file_lead) :])
    return ("add", file_lead)


def _apply_indent_to_line(line: str, kind: str, ws: str) -> str:
    if not ws or not _rstrip_nl(line).strip():
        return line
    ending = line[len(_rstrip_nl(line)) :]
    body = _rstrip_nl(line)
    if kind == "add":
        return ws + body + ending
    if body.startswith(ws):
        return body[len(ws) :] + ending
    lead = _leading_ws(line)
    return body[len(lead) :] + ending


def _adjust_new_string(new_string: str, hit: EditHit) -> str:
    if not hit.indent_ws:
        return _adapt_newlines(new_string, hit.matched)
    lines = _file_lines(new_string)
    shifted = [_apply_indent_to_line(ln, hit.indent_kind, hit.indent_ws) for ln in lines]
    adjusted = "".join(shifted)
    return _adapt_newlines(adjusted, hit.matched)


def _adapt_newlines(new_string: str, matched: str) -> str:
    if "\r\n" in matched and "\r\n" not in new_string:
        return new_string.replace("\n", "\r\n")
    if "\r\n" not in matched and "\r\n" in new_string:
        return new_string.replace("\r\n", "\n")
    return new_string


def _replace_hits(text: str, hits: list[EditHit], new_string: str) -> str:
    ordered = sorted(hits, key=lambda h: h.start)
    pieces: list[str] = []
    last = 0
    for hit in ordered:
        pieces.append(text[last : hit.start])
        pieces.append(_adjust_new_string(new_string, hit))
        last = hit.end
    pieces.append(text[last:])
    return "".join(pieces)


def _fuzzy_min_ratio(nonempty_lines: int, needle_chars: int) -> float:
    if nonempty_lines <= 1:
        if needle_chars < 24:
            return 0.92
        return 0.88
    if nonempty_lines <= 3:
        return 0.82
    return 0.76


def _line_match_ratio(left: str, right: str) -> float:
    a = _norm_whitespace(left)
    b = _norm_whitespace(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _fuzzy_scan_indices(file_lines: list[str], needle_lines: list[str], window: int) -> list[int]:
    max_i = max(1, len(file_lines) - window + 1)
    anchor_raw = ""
    for line in needle_lines:
        body = _rstrip_nl(line).strip()
        if body:
            anchor_raw = body[:80]
            break
    if len(file_lines) <= _FUZZY_FULL_SCAN_LINES:
        return list(range(max_i))
    if not anchor_raw:
        return list(range(0, max_i, max(1, max_i // 400)))
    prefix = anchor_raw[:24]
    hits = [
        i
        for i in range(max_i)
        if prefix in file_lines[i] or _collapse_ws(prefix) in _collapse_ws(file_lines[i])
    ]
    if hits:
        return hits
    return list(range(0, max_i, max(1, max_i // 400)))


def _rank_fuzzy_windows(text: str, needle: str) -> list[EditHit]:
    needle_lines = _file_lines(needle)
    file_lines = _file_lines(text)
    window = len(needle_lines)
    if window <= 0 or window > _FUZZY_MAX_NEEDLE_LINES or window > len(file_lines):
        return []
    nonempty = sum(1 for ln in needle_lines if _rstrip_nl(ln).strip())
    if nonempty <= 0:
        return []
    if nonempty == 1 and len(_norm_whitespace(needle)) < _FUZZY_MIN_SINGLE_LINE_CHARS:
        return []
    n_idx = _first_nonempty_index(needle_lines)
    needle_join = "\n".join(_norm_trailing_ws(ln) for ln in needle_lines)
    min_ratio = _fuzzy_min_ratio(nonempty, len(needle_join))
    offsets = _line_offsets(file_lines)
    ranked: list[EditHit] = []
    for i in _fuzzy_scan_indices(file_lines, needle_lines, window):
        chunk_lines = file_lines[i : i + window]
        if len(chunk_lines) < window:
            continue
        first_ratio = _line_match_ratio(needle_lines[n_idx], chunk_lines[min(n_idx, len(chunk_lines) - 1)])
        if first_ratio < _FUZZY_MIN_FIRST_LINE_RATIO:
            continue
        chunk_join = "\n".join(_norm_trailing_ws(ln) for ln in chunk_lines)
        quick = SequenceMatcher(None, needle_join, chunk_join, autojunk=False).quick_ratio()
        if quick < min_ratio - 0.08:
            continue
        ratio = SequenceMatcher(None, needle_join, chunk_join, autojunk=False).ratio()
        if ratio < 0.35:
            continue
        start = offsets[i]
        end = offsets[i + window]
        if not needle.endswith(("\n", "\r")):
            last = chunk_lines[-1]
            end -= len(last) - len(_rstrip_nl(last))
        file_lead = _leading_ws(chunk_lines[min(n_idx, len(chunk_lines) - 1)])
        needle_lead = _leading_ws(needle_lines[n_idx])
        kind, ws = _indent_delta(file_lead, needle_lead)
        ranked.append(
            EditHit(
                start=start,
                end=end,
                matched=text[start:end],
                strategy="fuzzy",
                indent_kind=kind,
                indent_ws=ws,
                ratio=ratio,
            )
        )
    ranked.sort(key=lambda hit: hit.ratio, reverse=True)
    return ranked


def find_fuzzy_unique_hit(text: str, needle: str) -> EditHit | None:
    """
    精确族失败后：若存在唯一高相似行窗则返回该命中。

    门槛：相似度过线，且第一行也够像；若有第二候选，差距须 ≥ 0.12。
    replace_all 场景不要调用。

    @param text 文件全文
    @param needle old_string（可已剥行号前缀）
    @return 唯一 fuzzy 命中；不够自信则为 None
    """
    cleaned = strip_read_file_artifacts(needle)
    ranked = _rank_fuzzy_windows(text, cleaned)
    if not ranked:
        return None
    nonempty = sum(1 for ln in cleaned.splitlines() if ln.strip())
    min_ratio = _fuzzy_min_ratio(nonempty, len(cleaned.strip()))
    best = ranked[0]
    if best.ratio < min_ratio:
        return None
    if len(ranked) > 1 and (best.ratio - ranked[1].ratio) < _FUZZY_MIN_GAP:
        return None
    return best


def _best_window_hint(text: str, needle: str, *, limit: int) -> str:
    ranked = _rank_fuzzy_windows(text, needle)
    if not ranked:
        return ""
    best = ranked[0]
    if best.ratio < 0.35:
        return ""
    file_lines = text.splitlines()
    start_line = text[: best.start].count("\n")
    window = max(1, min(limit, len(_file_lines(needle)) or 1))
    end = min(len(file_lines), start_line + window)
    shown = [
        f"  L{start_line + j + 1}: {file_lines[start_line + j][:120]}"
        for j in range(end - start_line)
        if start_line + j < len(file_lines)
    ]
    if not shown:
        return ""
    return f"最相近片段（相似度 {best.ratio:.0%}）:\n" + "\n".join(shown)


def parse_replacements_arg(raw: object) -> list[EditHunk]:
    """
    解析模型传入的 replacements 列表。

    @param raw 工具参数
    @return hunk 列表；无法解析则为空
    """
    if not raw:
        return []
    if not isinstance(raw, list):
        return []
    out: list[EditHunk] = []
    for item in raw:
        if not isinstance(item, dict):
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            else:
                continue
        if not isinstance(item, dict):
            continue
        old = item.get("old_string")
        if old is None:
            old = item.get("oldString")
        if not isinstance(old, str):
            continue
        new = item.get("new_string")
        if new is None:
            new = item.get("newString")
        if new is None:
            new = ""
        replace_all = bool(item.get("replace_all") or item.get("replaceAll"))
        out.append(EditHunk(old_string=old, new_string=str(new), replace_all=replace_all))
    return out
