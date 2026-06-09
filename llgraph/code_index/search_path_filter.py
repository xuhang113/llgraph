"""检索结果路径过滤（排除索引/遍历噪声）。"""

from __future__ import annotations

_JUNK_PATH_MARKERS: tuple[str, ...] = (
    "/.git/",
    "/node_modules/",
    "/target/",
    "/dist/",
    "/build/",
    "/.venv/",
    "/venv/",
    "/__pycache__/",
)

_JUNK_BASENAMES: frozenset[str] = frozenset({
    ".DS_Store",
    "Thumbs.db",
})


def is_junk_search_path(rel_path: str) -> bool:
    """
    判断路径是否应排除在检索结果外。

    @param rel_path 工作区相对路径
    @return 是否为噪声路径
    """
    if not rel_path or not rel_path.strip():
        return True
    rel = rel_path.replace("\\", "/").strip()
    base = rel.rsplit("/", 1)[-1]
    if base in _JUNK_BASENAMES:
        return True
    if rel.endswith(".DS_Store"):
        return True
    for marker in _JUNK_PATH_MARKERS:
        if marker in f"/{rel}/":
            return True
    return False
