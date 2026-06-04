"""行窗口代码切块（Phase 2a）。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from llgraph.code_index.file_scan import language_from_path
from llgraph.code_index.paths import (
    CHUNK_OVERLAP_LINES,
    CHUNK_TARGET_CHARS,
    MAX_FILE_BYTES,
    TEXT_PREVIEW_MAX,
)


@dataclass(frozen=True)
class CodeChunk:
    """单个代码块。"""

    chunk_id: str
    rel_path: str
    start_line: int
    end_line: int
    language: str
    symbol: str
    content_hash: str
    text: str

    @property
    def text_preview(self) -> str:
        preview = self.text.replace("\n", " ").strip()
        if len(preview) > TEXT_PREVIEW_MAX:
            return preview[: TEXT_PREVIEW_MAX - 1] + "…"
        return preview


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_chunk_id(rel_path: str, start: int, end: int, content_hash: str) -> str:
    return f"{rel_path}:{start}:{end}:{content_hash[:16]}"


def chunk_file_text(
    rel_path: str,
    text: str,
    *,
    use_ast: bool = False,
) -> list[CodeChunk]:
    """
    将文件文本切分为 chunk 列表。

    @param rel_path 相对路径
    @param text 文件全文
    @param use_ast 为 True 时尝试 AST 切块（失败则回退行窗口）
    @return CodeChunk 列表
    """
    if use_ast:
        from llgraph.code_index.ast_chunker import chunk_file_ast

        ast_chunks = chunk_file_ast(rel_path, text)
        if ast_chunks:
            return ast_chunks

    lines = text.splitlines()
    if not lines:
        return []

    language = language_from_path(rel_path)
    chunks: list[CodeChunk] = []
    start_idx = 0
    n = len(lines)

    while start_idx < n:
        char_count = 0
        end_idx = start_idx
        while end_idx < n and char_count < CHUNK_TARGET_CHARS:
            char_count += len(lines[end_idx]) + 1
            end_idx += 1

        if end_idx == start_idx:
            end_idx = min(start_idx + 1, n)

        block_lines = lines[start_idx:end_idx]
        block_text = "\n".join(block_lines)
        if block_text.strip():
            chash = _content_hash(block_text)
            start_line = start_idx + 1
            end_line = end_idx
            chunks.append(
                CodeChunk(
                    chunk_id=_make_chunk_id(rel_path, start_line, end_line, chash),
                    rel_path=rel_path,
                    start_line=start_line,
                    end_line=end_line,
                    language=language,
                    symbol="",
                    content_hash=chash,
                    text=block_text,
                )
            )

        if end_idx >= n:
            break
        start_idx = max(end_idx - CHUNK_OVERLAP_LINES, start_idx + 1)

    return chunks


def chunk_file_path(rel_path: str, full_path: Path, *, use_ast: bool = False) -> list[CodeChunk]:
    """
    读取文件并切块；过大文件截断或跳过。

    @param rel_path 相对路径
    @param full_path 绝对路径
    @param use_ast 是否尝试 AST
    @return chunk 列表；跳过时返回空列表
    """
    try:
        size = full_path.stat().st_size
    except OSError:
        return []

    if size > MAX_FILE_BYTES:
        try:
            raw = full_path.read_bytes()[:MAX_FILE_BYTES]
            text = raw.decode("utf-8", errors="replace")
        except OSError:
            return []
    else:
        try:
            text = full_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

    return chunk_file_text(rel_path, text, use_ast=use_ast)
