"""从检索工具结果抽取 path:line 命中，供 read spill 命中区预览。"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.messages import BaseMessage, ToolMessage

_READ_RANGE_HDR = re.compile(
    r"^---\s+(.+?)\s+\(行\s+(\d+)-(\d+)\s+/ 共\s+\d+\s+行\)",
    re.MULTILINE,
)
# grep / 格式化块：path:line: 或 --- path:line ---
_GREP_HIT = re.compile(r"^---\s+(.+?):(\d+)\s+---", re.MULTILINE)
_GREP_LINE_HIT = re.compile(r"^([^:\n]+?):(\d+):\s", re.MULTILINE)
# parallel / semantic：1. path:42  snippet
_PARALLEL_HIT = re.compile(r"^\d+\.\s+(.+?):(\d+)(?:\s|$|-)", re.MULTILINE)

_SEARCH_TOOLS = frozenset(
    {
        "grep_files",
        "search_code_parallel",
        "search_code_semantic",
        "search_files",
    }
)


def extract_read_source_paths(content: str) -> list[str]:
    """@param content read 工具输出 @return 源文件相对路径列表"""
    paths: list[str] = []
    seen: set[str] = set()
    for match in _READ_RANGE_HDR.finditer(content):
        rel = match.group(1).strip()
        if rel and rel not in seen:
            seen.add(rel)
            paths.append(rel)
    return paths


def parse_search_hits_from_content(content: str) -> dict[str, list[int]]:
    """
    从工具输出文本解析 path → 行号列表。

    @param content grep / parallel 等工具返回
    @return 相对路径到行号（升序去重）
    """
    by_path: dict[str, set[int]] = {}
    for pattern in (_GREP_HIT, _GREP_LINE_HIT, _PARALLEL_HIT):
        for match in pattern.finditer(content):
            rel = match.group(1).strip()
            try:
                line_no = int(match.group(2))
            except ValueError:
                continue
            if not rel or line_no <= 0:
                continue
            by_path.setdefault(rel, set()).add(line_no)
    return {path: sorted(lines) for path, lines in by_path.items()}


def collect_search_hits_from_messages(
    messages: list[BaseMessage],
) -> dict[str, list[int]]:
    """
    汇总消息历史中检索类工具记录的命中行。

    @param messages 图状态 messages
    @return path → 行号列表
    """
    merged: dict[str, set[int]] = {}
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        name = str(getattr(msg, "name", "") or "")
        if name not in _SEARCH_TOOLS:
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        for path, lines in parse_search_hits_from_content(content).items():
            merged.setdefault(path, set()).update(lines)
    return {path: sorted(lines) for path, lines in merged.items()}


def _merge_line_ranges(
    ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """@param ranges (start,end) 含端点 @return 合并后区间"""
    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda r: (r[0], r[1]))
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def read_source_line_window(
    workspace: Path,
    rel_path: str,
    center_line: int,
    radius: int,
) -> str | None:
    """
    从工作区源文件读取 center_line ± radius 行（带行号）。

    @param workspace 工作区根
    @param rel_path 相对路径
    @param center_line 中心行号（1-based）
    @param radius 上下扩展行数
    @return 格式化文本；失败 None
    """
    from llgraph.config.catalog_paths import resolve_catalog_read_path

    try:
        target = resolve_catalog_read_path(workspace, rel_path, sandbox=None)
    except ValueError:
        return None
    if not target.is_file():
        return None
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = text.splitlines()
    if not lines:
        return None
    start = max(1, center_line - radius)
    end = min(len(lines), center_line + radius)
    body = "\n".join(
        f"{start + i}| {line}" for i, line in enumerate(lines[start - 1 : end])
    )
    return f"--- {rel_path} (行 {start}-{end} / 共 {len(lines)} 行) ---\n{body}"


def build_hit_anchor_preview(
    workspace: Path,
    *,
    source_paths: set[str],
    hits_by_path: dict[str, list[int]],
    radius: int,
    max_chars: int = 24_000,
) -> str:
    """
    为 read spill 生成「历史检索命中 ±radius」预览。

    @param workspace 工作区根
    @param source_paths read 内容涉及的源文件路径
    @param hits_by_path 历史命中
    @param radius 上下行数
    @param max_chars 预览总字符上限
    @return 空串或预览块
    """
    if not source_paths or not hits_by_path or radius <= 0:
        return ""

    blocks: list[str] = []
    total = 0
    for path in sorted(source_paths):
        line_nos = hits_by_path.get(path)
        if not line_nos:
            continue
        ranges = _merge_line_ranges(
            [
                (max(1, ln - radius), ln + radius)
                for ln in line_nos[:12]
            ]
        )
        for start, end in ranges:
            center = (start + end) // 2
            block = read_source_line_window(
                workspace,
                path,
                center_line=center,
                radius=max(center - start, end - center),
            )
            if not block:
                continue
            if total + len(block) > max_chars:
                blocks.append(
                    f"（命中区预览已截断，剩余 {len(line_nos)} 处命中请 read_file 按需读取）"
                )
                return "\n\n".join(blocks)
            blocks.append(block)
            total += len(block) + 2

    return "\n\n".join(blocks)
