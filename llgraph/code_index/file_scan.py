"""可索引文件扫描（复用 workspace 跳过规则）。"""

import hashlib
import os
from pathlib import Path

from llgraph.core.filesystem_tools import _is_probably_text
from llgraph.core.workspace import WorkspaceContext

# 历史默认上限；现默认 0=不限制，见 embedding.json 的 index.max_files
MAX_INDEX_FILES = 0

_INDEX_SUFFIXES = frozenset({
    ".py", ".java", ".kt", ".go", ".rs", ".js", ".ts", ".tsx", ".jsx",
    ".md", ".txt", ".yaml", ".yml", ".json", ".xml", ".properties",
    ".sql", ".sh", ".zsh", ".toml", ".ini", ".cfg", ".html", ".css",
    ".vue", ".gradle", ".mdc",
})


def language_from_path(rel_path: str) -> str:
    """
    从路径推断语言标识。

    @param rel_path 相对路径
    @return 语言名或空
    """
    suffix = Path(rel_path).suffix.lower()
    mapping = {
        ".py": "python",
        ".java": "java",
        ".kt": "kotlin",
        ".go": "go",
        ".rs": "rust",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".md": "markdown",
        ".xml": "xml",
        ".sql": "sql",
    }
    return mapping.get(suffix, suffix.lstrip(".") if suffix else "")


def file_sha256(full_path: Path) -> str:
    """
    计算文件 SHA256。

    @param full_path 绝对路径
    @return 十六进制摘要
    """
    digest = hashlib.sha256()
    with full_path.open("rb") as handle:
        while True:
            block = handle.read(65536)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def iter_indexable_files(
    ctx: WorkspaceContext,
    *,
    path_prefix: str = ".",
    max_files: int | None = MAX_INDEX_FILES,
):
    """
    遍历可索引文本文件。

    @param ctx 工作区上下文
    @param path_prefix 起始相对目录
    @param max_files 最多文件数；None 或 0 表示不限制
    @yield (rel_path, full_path) 元组
    """
    base = ctx.resolve_path(path_prefix)
    if not base.is_dir():
        return

    limit = None if not max_files else max_files
    count = 0
    truncated = False
    for dirpath, dirnames, filenames in os.walk(base, topdown=True):
        dirnames[:] = [d for d in dirnames if not ctx.should_skip_dir(d)]
        if str(dirpath).count(os.sep) - str(base).count(os.sep) > 12:
            dirnames.clear()
        for filename in filenames:
            full = Path(dirpath) / filename
            if not full.is_file():
                continue
            if not _is_probably_text(full) and full.suffix.lower() not in _INDEX_SUFFIXES:
                continue
            try:
                rel = full.relative_to(ctx.root).as_posix()
            except ValueError:
                continue
            yield rel, full
            count += 1
            if limit is not None and count >= limit:
                return
