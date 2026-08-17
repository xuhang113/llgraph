"""检索类工具结果内联截断（grep 等不落盘，超长时保留命中块预览）。"""

from __future__ import annotations

import re

_SEARCH_TOOLS = frozenset({
    "grep_files",
    "search_code_parallel",
    "search_code_semantic",
    "search_files",
    "search_workspace",
})

_GREP_BLOCK_HDR = re.compile(r"^---\s+(.+?):(\d+)\s+---\s*$", re.MULTILINE)
_DEFAULT_MAX_INLINE_CHARS = 48_000
_DEFAULT_MAX_BLOCKS = 12


def is_search_tool(tool_name: str) -> bool:
    """@param tool_name 工具名 @return 是否为检索类工具"""
    return str(tool_name or "").strip() in _SEARCH_TOOLS


def split_search_hit_blocks(content: str) -> list[str]:
    """
    将 grep 风格输出拆成命中块。

    @param content 工具输出
    @return 块列表（含 header 行）
    """
    if not content.strip():
        return []
    parts = _GREP_BLOCK_HDR.split(content)
    if len(parts) <= 1:
        return [content.strip()] if content.strip() else []
    blocks: list[str] = []
    prefix = parts[0].strip()
    idx = 1
    while idx + 2 <= len(parts):
        path = parts[idx].strip()
        line_no = parts[idx + 1].strip()
        body = parts[idx + 2].strip()
        block = f"--- {path}:{line_no} ---"
        if body:
            block = f"{block}\n{body}"
        blocks.append(block)
        idx += 3
    if prefix:
        blocks.insert(0, prefix)
    return blocks


def _score_search_block(block: str) -> tuple[int, int]:
    """命中块排序：含 >>> 优先，其次块长度（短优先保留多样性）。"""
    has_hit = 1 if ">>>" in block else 0
    return (has_hit, -len(block))


def clip_search_tool_result(
    tool_name: str,
    content: str,
    *,
    max_chars: int = _DEFAULT_MAX_INLINE_CHARS,
    max_blocks: int = _DEFAULT_MAX_BLOCKS,
) -> str:
    """
    检索工具结果内联截断（不落盘）。

    @param tool_name 工具名
    @param content 原始输出
    @param max_chars 内联字符上限
    @param max_blocks 截断时最多保留命中块数
    @return 可能截断后的文本
    """
    if not is_search_tool(tool_name) or len(content) <= max_chars:
        return content
    if "未找到匹配" in content or content.strip().startswith("错误"):
        return content

    blocks = split_search_hit_blocks(content)
    if not blocks:
        return content[: max_chars - 80] + "\n…（检索结果已内联截断，请缩小 pattern 或 path）"

    header = blocks[0] if not blocks[0].startswith("---") else ""
    hit_blocks = blocks if not header else blocks[1:]
    if not hit_blocks:
        return content[: max_chars - 80] + "\n…（检索结果已内联截断）"

    ranked = sorted(hit_blocks, key=_score_search_block, reverse=True)
    kept: list[str] = []
    total = len(header) + 1 if header else 0
    for block in ranked:
        extra = len(block) + (2 if kept else 0)
        if total + extra > max_chars and kept:
            break
        if len(kept) >= max_blocks:
            break
        kept.append(block)
        total += extra

    if not kept:
        kept = [ranked[0][: max_chars - 120]]

    parts: list[str] = []
    if header:
        parts.append(header)
    parts.extend(kept)
    omitted = len(hit_blocks) - len(kept)
    if omitted > 0:
        parts.append(
            f"…（内联截断：原 {len(hit_blocks)} 个命中块，已保留 {len(kept)} 个；"
            f"请缩小 pattern/path 或 read_file 查看具体文件）"
        )
    return "\n\n".join(parts)
