"""Code index 目录与常量。"""

from pathlib import Path

LLGRAPH_DIR = ".llgraph"
INDEX_DIR_NAME = "index"
LANCE_SUBDIR = "lance"
TABLE_NAME = "code_chunks"
MANIFEST_FILENAME = "manifest.json"
EMBED_CACHE_FILENAME = "embed_cache.db"
META_FILENAME = "index_meta.json"

DEFAULT_VECTOR_DIM = 1536
TEXT_PREVIEW_MAX = 200

# 行窗口切块（约 400–600 token，按字符近似）
CHUNK_TARGET_CHARS = 2400
CHUNK_OVERLAP_LINES = 50
MAX_FILE_BYTES = 500_000


def index_root(workspace: Path) -> Path:
    """工作区索引根：.llgraph/index。"""
    return workspace.expanduser().resolve() / LLGRAPH_DIR / INDEX_DIR_NAME


def lance_uri(workspace: Path) -> str:
    """LanceDB 连接 URI（本地目录）。"""
    return str(index_root(workspace) / LANCE_SUBDIR)


def manifest_path(workspace: Path) -> Path:
    """manifest.json 路径。"""
    return index_root(workspace) / MANIFEST_FILENAME


def embed_cache_path(workspace: Path) -> Path:
    """embedding SQLite 缓存路径。"""
    return index_root(workspace) / EMBED_CACHE_FILENAME


def meta_path(workspace: Path) -> Path:
    """索引元信息路径。"""
    return index_root(workspace) / META_FILENAME


def ensure_index_dirs(workspace: Path) -> Path:
    """
    创建索引目录。

    @param workspace 工作区根
    @return index 根路径
    """
    root = index_root(workspace)
    root.mkdir(parents=True, exist_ok=True)
    (root / LANCE_SUBDIR).mkdir(parents=True, exist_ok=True)
    return root
