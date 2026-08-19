"""search_replace 容错匹配：对齐 Cursor / Claude Code / Codex 的改码命中路径。

优先 exact；失败则依次尝试换行符、读文件行号前缀、行尾空白、缩进对齐、行内空白折叠。
同一文件多个 hunk 按顺序作用在内存文本上，全部成功后再落盘。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

_READ_LINE_PREFIX = re.compile(r"^\d+\| ")
_READ_HEADER = re.compile(
    r"^---\s+.+\s+\(行\s+\d+-\d+\s+/ 共\s+\d+\s+行\)(?:\s+---)?\s*\n?",
)
_WS_RUN = re.compile(r"[^\S\n]+")

_STRATEGY_ORDER = (
    "exact",
    "newline",
    "trailing_ws",
    "indent",
    "whitespace",
)


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


def apply_search_replace(
    text: str,
    old_string: str,
    new_string: str,
    *,
    replace_all: bool = False,
    require_unique: bool = True,
) -> ApplyEditResult:
    """
    对单段文本做一次容错替换。

    @param text 当前文件内容
    @param old_string 待匹配片段
    @param new_string 替换文本
    @param replace_all 是否替换全部命中
    @param require_unique 非 replace_all 时是否要求唯一命中
    @return 应用结果（失败时不改原文）
    """
    return apply_edit_hunks(
        text,
        [EditHunk(old_string, new_string, replace_all=replace_all)],
        require_unique=require_unique,
    )


def apply_edit_hunks(
    text: str,
    hunks: list[EditHunk],
    *,
    require_unique: bool = True,
) -> ApplyEditResult:
    """
    按顺序应用多个 hunk；任一失败则整体失败。

    @param text 当前文件内容
    @param hunks 替换列表
    @param require_unique 非 replace_all 时是否要求唯一命中
    @return 应用结果
    """
    cleaned = [h for h in hunks if (h.old_string or "").strip() or h.old_string == ""]
    if not cleaned:
        return ApplyEditResult(ok=False, error="必须提供 old_string 或 replacements")
    current = text
    total = 0
    strategies: list[str] = []
    all_tried: list[str] = []
    for idx, hunk in enumerate(cleaned, start=1):
        one = _apply_one_hunk(current, hunk, require_unique=require_unique)
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
    strategy = strategies[0] if len(set(strategies)) == 1 else "+".join(strategies)
    return ApplyEditResult(
        ok=True,
        new_text=current,
        replacements=total,
        hunks_applied=len(cleaned),
        strategy=strategy,
        tried=tuple(dict.fromkeys(all_tried)),
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


def format_apply_success(rel: str, result: ApplyEditResult) -> str:
    """
    成功替换后的工具返回文案。

    @param rel 相对路径
    @param result 成功结果
    @return 工具成功文本
    """
    hunk_bit = ""
    if result.hunks_applied > 1:
        hunk_bit = f" / {result.hunks_applied} hunk"
    strategy_bit = ""
    if result.strategy and result.strategy != "exact" and "exact" not in result.strategy.split("+"):
        strategy_bit = f"，匹配策略={result.strategy}"
    elif result.strategy and "+" in result.strategy and result.strategy != "exact":
        parts = result.strategy.split("+")
        if any(p != "exact" for p in parts):
            strategy_bit = f"，匹配策略={result.strategy}"
    return f"已替换 {rel}（{result.replacements} 处{hunk_bit}{strategy_bit}）"


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
) -> ApplyEditResult:
    needle = hunk.old_string
    if needle == "":
        return ApplyEditResult(ok=False, error="错误: search_replace 缺少 old_string")
    needles = [needle]
    stripped = strip_read_file_artifacts(needle)
    if stripped != needle:
        needles.append(stripped)

    tried: list[str] = []
    last_hits: list[EditHit] = []
    for candidate in needles:
        for strategy in _STRATEGY_ORDER:
            tried.append(strategy)
            hits = _find_hits(text, candidate, strategy)
            if not hits:
                continue
            last_hits = hits
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
    if hit.strategy != "indent" or not hit.indent_ws:
        return _adapt_newlines(new_string, hit.matched)
    lines = _file_lines(new_string)
    shifted = [_apply_indent_to_line(ln, hit.indent_kind, hit.indent_ws) for ln in lines]
    adjusted = "".join(shifted) if new_string.endswith(("\n", "\r")) else "".join(shifted)
    if not new_string.endswith(("\n", "\r")) and shifted:
        # join via original splitlines keepends already includes endings except possibly last
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


def _best_window_hint(text: str, needle: str, *, limit: int) -> str:
    needle_lines = [ln.rstrip() for ln in needle.splitlines() if ln.strip()]
    if not needle_lines:
        return ""
    file_lines = text.splitlines()
    window = max(len(needle.splitlines()), 1)
    if window > 40:
        window = 40
    needle_join = "\n".join(needle_lines[:window])
    best_ratio = 0.0
    best_i = 0
    max_i = max(1, len(file_lines) - window + 1)
    # 大文件：只在首行近似命中附近扫描
    scan_range: list[int]
    anchor = needle_lines[0][:80] if needle_lines else ""
    if len(file_lines) > 8000 and anchor:
        scan_range = [
            i
            for i in range(max_i)
            if anchor[:24] in file_lines[i] or _collapse_ws(anchor[:24]) in _collapse_ws(file_lines[i])
        ]
        if not scan_range:
            scan_range = list(range(0, max_i, max(1, max_i // 400)))
    else:
        scan_range = list(range(max_i))
    for i in scan_range:
        chunk = "\n".join(ln.rstrip() for ln in file_lines[i : i + window])
        ratio = SequenceMatcher(None, needle_join, chunk).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_i = i
    if best_ratio < 0.35:
        return ""
    end = min(len(file_lines), best_i + min(window, limit))
    shown = [
        f"  L{best_i + j + 1}: {file_lines[best_i + j][:120]}"
        for j in range(end - best_i)
    ]
    return f"最相近片段（相似度 {best_ratio:.0%}）:\n" + "\n".join(shown)


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
