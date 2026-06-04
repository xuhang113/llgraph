"""AST 切块（Phase 2b，tree-sitter 可选）。"""

from __future__ import annotations

import hashlib
from pathlib import Path

from llgraph.code_index.chunker import CodeChunk, _content_hash, _make_chunk_id
from llgraph.code_index.file_scan import language_from_path
from llgraph.code_index.paths import CHUNK_TARGET_CHARS


def _tree_sitter_available() -> bool:
    try:
        import tree_sitter  # noqa: F401
        return True
    except ImportError:
        return False


def chunk_file_ast(rel_path: str, text: str) -> list[CodeChunk]:
    """
    使用 tree-sitter 按函数/类切块；不可用或解析失败时返回空列表。

    @param rel_path 相对路径
    @param text 源文件内容
    @return CodeChunk 列表
    """
    if not _tree_sitter_available():
        return []

    suffix = Path(rel_path).suffix.lower()
    if suffix not in (".py", ".java", ".js", ".ts", ".tsx", ".go"):
        return []

    try:
        return _chunk_with_tree_sitter(rel_path, text, suffix)
    except Exception:
        return []


def _chunk_with_tree_sitter(rel_path: str, text: str, suffix: str) -> list[CodeChunk]:
    from tree_sitter import Language, Parser
    import tree_sitter_python as tspython
    import tree_sitter_java as tsjava
    import tree_sitter_javascript as tsjavascript

    if suffix == ".py":
        lang = Language(tspython.language())
    elif suffix == ".java":
        lang = Language(tsjava.language())
    elif suffix in (".js", ".ts", ".tsx"):
        lang = Language(tsjavascript.language())
    else:
        return []

    parser = Parser(lang)
    tree = parser.parse(text.encode("utf-8"))
    if tree.root_node.has_error:
        return []

    node_types = {
        "python": ("function_definition", "class_definition"),
        "java": ("method_declaration", "class_declaration"),
        "javascript": ("function_declaration", "class_declaration", "method_definition"),
    }
    lang_key = "python" if suffix == ".py" else ("java" if suffix == ".java" else "javascript")
    targets = node_types.get(lang_key, ())

    lines = text.splitlines()
    language = language_from_path(rel_path)
    chunks: list[CodeChunk] = []

    def visit(node):
        if node.type in targets:
            start_row = node.start_point[0]
            end_row = node.end_point[0]
            block_lines = lines[start_row : end_row + 1]
            block_text = "\n".join(block_lines)
            if len(block_text.strip()) < 20:
                return
            if len(block_text) > CHUNK_TARGET_CHARS * 2:
                return
            chash = _content_hash(block_text)
            start_line = start_row + 1
            end_line = end_row + 1
            symbol = node.type
            chunks.append(
                CodeChunk(
                    chunk_id=_make_chunk_id(rel_path, start_line, end_line, chash),
                    rel_path=rel_path,
                    start_line=start_line,
                    end_line=end_line,
                    language=language,
                    symbol=symbol,
                    content_hash=chash,
                    text=block_text,
                )
            )
        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return chunks
