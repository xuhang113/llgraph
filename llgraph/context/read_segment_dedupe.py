"""read 工具出站去重：按路径 + 行段，仅重叠 ≥50% 时替换。"""

from __future__ import annotations

import re

from langchain_core.messages import ToolMessage

_READ_RANGE_HDR = re.compile(
    r"^---\s+(.+?)\s+\(行\s+(\d+)-(\d+)\s+/ 共\s+\d+\s+行\)",
    re.MULTILINE,
)

OVERLAP_SUPERSEDE_RATIO = 0.5


def extract_read_segments(content: str) -> list[tuple[str, int, int]]:
    """
    @param content read_file / read_files 输出
    @return [(相对路径, start_line, end_line), ...]
    """
    segments: list[tuple[str, int, int]] = []
    for match in _READ_RANGE_HDR.finditer(content):
        path = match.group(1).strip()
        try:
            start = int(match.group(2))
            end = int(match.group(3))
        except ValueError:
            continue
        if path and start > 0 and end >= start:
            segments.append((path, start, end))
    return segments


def segment_overlap_ratio(
    a_start: int,
    a_end: int,
    b_start: int,
    b_end: int,
) -> float:
    """@return a 区间被 b 覆盖的比例（0~1）"""
    overlap = max(0, min(a_end, b_end) - max(a_start, b_start) + 1)
    if overlap <= 0:
        return 0.0
    len_a = a_end - a_start + 1
    return overlap / len_a


def segment_superseded_by_later(
    seg: tuple[str, int, int],
    later_segments: list[tuple[str, int, int]],
    *,
    threshold: float = OVERLAP_SUPERSEDE_RATIO,
) -> bool:
    """@return 是否存在更晚 read 与 seg 重叠 ≥ threshold"""
    path, start, end = seg
    for lp, ls, le in later_segments:
        if lp != path:
            continue
        if segment_overlap_ratio(start, end, ls, le) >= threshold:
            return True
    return False


def format_superseded_read_pointer(segments: list[tuple[str, int, int]]) -> str:
    """@param segments 被替换的行段 @return 短指针文案"""
    if len(segments) == 1:
        path, start, end = segments[0]
        hint = f"`{path}` 行 {start}-{end}"
    else:
        path, start, end = segments[0]
        hint = f"`{path}` 行 {start}-{end} 等 {len(segments)} 段"
    return (
        f"[历史 read 已替换] {hint} 与后续 read 行段重叠 ≥50%；"
        f"互补行段仍保留；需要其它行段时用 start_line/end_line。"
    )


def read_message_fully_superseded(
    msg_idx: int,
    segments: list[tuple[str, int, int]],
    all_later_segments: list[tuple[str, int, int]],
) -> bool:
    """
    @param msg_idx 消息索引（保留供扩展）
    @param segments 本消息行段
    @param all_later_segments 更晚消息的全部行段
    @return 是否所有行段均被更晚 read 重叠替换
    """
    _ = msg_idx
    if not segments:
        return False
    return all(
        segment_superseded_by_later(seg, all_later_segments) for seg in segments
    )
