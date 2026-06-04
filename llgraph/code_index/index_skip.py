"""索引跳过目录：扫描排除与已入库数据清理。"""

from __future__ import annotations

from pathlib import Path

from llgraph.code_index.store import delete_chunks_under_top_dir


def rel_under_skip_dir(rel_path: str, skip_dirs: frozenset[str]) -> bool:
    """
    相对路径是否位于 skip_dirs 中的顶层目录下。

    @param rel_path 工作区相对路径
    @param skip_dirs 要跳过的顶层目录名集合
    @return 是否应忽略
    """
    if not skip_dirs:
        return False
    top = rel_path.split("/", 1)[0]
    return top in skip_dirs


def purge_skipped_index_entries(
    workspace: Path,
    manifest: dict[str, str],
    skip_dirs: frozenset[str],
) -> int:
    """
    从 manifest 与 Lance 中移除 skip_dirs 下已有索引。

    @param workspace 工作区根
    @param manifest 当前 manifest（就地修改）
    @param skip_dirs 跳过的顶层目录名
    @return 移除的文件数
    """
    if not skip_dirs:
        return 0
    removed = 0
    for rel in list(manifest.keys()):
        if not rel_under_skip_dir(rel, skip_dirs):
            continue
        del manifest[rel]
        removed += 1
    for top_dir in sorted(skip_dirs):
        delete_chunks_under_top_dir(workspace, top_dir)
    return removed
