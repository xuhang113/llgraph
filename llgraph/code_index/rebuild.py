"""重建索引：清理旧数据后全量重扫。"""

from __future__ import annotations

from pathlib import Path

from llgraph.code_index.embedder import clear_embed_cache
from llgraph.code_index.index_logging import get_index_logger
from llgraph.code_index.manifest import clear_manifest, load_manifest, save_manifest
from llgraph.code_index.store import delete_chunks_for_file, drop_index_table


def _paths_under_prefix(manifest: dict[str, str], path_prefix: str) -> list[str]:
    prefix = path_prefix.strip().rstrip("/")
    if not prefix or prefix == ".":
        return list(manifest.keys())
    return [
        rel
        for rel in manifest
        if rel == prefix or rel.startswith(prefix + "/")
    ]


def prepare_rebuild(
    workspace: Path,
    *,
    path_prefix: str = ".",
    clear_embedding_cache: bool = False,
) -> None:
    """
    重建前清理：Lance 表、manifest、可选 embed 缓存。

    @param workspace 工作区根
    @param path_prefix 若不为 . 则仅清理该前缀下文件的 chunk 与 manifest 项
    @param clear_embedding_cache 是否删除 embed_cache.db
    """
    logger = get_index_logger()
    prefix = path_prefix.strip() or "."

    if prefix in (".", ""):
        logger.info("重建：删除 Lance 表 code_chunks")
        drop_index_table(workspace)
        logger.info("重建：清空 manifest.json")
        clear_manifest(workspace)
    else:
        logger.info("重建（子目录 %s）：删除相关 chunk", prefix)
        manifest = load_manifest(workspace)
        for rel in _paths_under_prefix(manifest, prefix):
            delete_chunks_for_file(workspace, rel)
        remaining = {
            k: v for k, v in manifest.items() if k not in _paths_under_prefix(manifest, prefix)
        }
        save_manifest(workspace, remaining)
        logger.info("重建：已从 manifest 移除 %d 条", len(manifest) - len(remaining))

    if clear_embedding_cache:
        if clear_embed_cache(workspace):
            logger.info("重建：已删除 embed_cache.db")
        else:
            logger.info("重建：无 embed_cache.db 可删")
