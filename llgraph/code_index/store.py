"""LanceDB 存储：code_chunks 表。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llgraph.code_index.paths import (
    DEFAULT_SEARCH_TOP_K,
    DEFAULT_VECTOR_DIM,
    META_FILENAME,
    TABLE_NAME,
    ensure_index_dirs,
    index_root,
    lance_uri,
    meta_path,
)


def _require_lancedb():
    try:
        import lancedb
    except ImportError as exc:
        raise RuntimeError(
            "未安装 lancedb。请执行: pip install 'llgraph[index]' 或 pip install lancedb"
        ) from exc
    return lancedb


@dataclass
class IndexStatus:
    """索引状态摘要。"""

    exists: bool
    chunk_count: int
    vector_dim: int
    last_indexed_at: str | None
    lance_path: str


def _load_meta(workspace: Path) -> dict[str, Any]:
    path = meta_path(workspace)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_meta(workspace: Path, meta: dict[str, Any]) -> None:
    ensure_index_dirs(workspace)
    meta_path(workspace).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def drop_index_table(workspace: Path) -> None:
    """删除 code_chunks 表（全量重建前）。"""
    db = connect_db(workspace)
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)


def connect_db(workspace: Path):
    """
    连接工作区 LanceDB。

    @param workspace 工作区根
    @return lancedb.DB connection
    """
    ensure_index_dirs(workspace)
    lancedb = _require_lancedb()
    return lancedb.connect(lance_uri(workspace))


def get_table(workspace: Path, *, vector_dim: int = DEFAULT_VECTOR_DIM):
    """
    打开或创建 code_chunks 表。

    @param workspace 工作区根
    @param vector_dim 向量维度
    @return LanceDB table
    """
    db = connect_db(workspace)
    names = db.table_names()
    if TABLE_NAME in names:
        return db.open_table(TABLE_NAME)
    # 空表：用占位行创建 schema 后立即删除
    placeholder = _chunk_record(
        chunk_id="__placeholder__",
        rel_path="",
        start_line=0,
        end_line=0,
        language="",
        symbol="",
        content_hash="",
        text_preview="",
        vector=[0.0] * vector_dim,
    )
    table = db.create_table(TABLE_NAME, data=[placeholder], mode="overwrite")
    table.delete('chunk_id = "__placeholder__"')
    meta = _load_meta(workspace)
    meta["vector_dim"] = vector_dim
    _save_meta(workspace, meta)
    return table


def _chunk_record(
    *,
    chunk_id: str,
    rel_path: str,
    start_line: int,
    end_line: int,
    language: str,
    symbol: str,
    content_hash: str,
    text_preview: str,
    vector: list[float],
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "rel_path": rel_path,
        "start_line": start_line,
        "end_line": end_line,
        "language": language,
        "symbol": symbol,
        "content_hash": content_hash,
        "text_preview": text_preview,
        "vector": vector,
    }


def upsert_chunks(workspace: Path, records: list[dict[str, Any]]) -> int:
    """
    批量写入 chunk 到 Lance。

    调用方须在写入前按 rel_path 删除旧数据（索引主路径在 flush 前已 delete_chunks_for_file）。
    禁止对每条 chunk_id 单独 delete，否则大文件（数百 chunk）会极慢。

    @param workspace 工作区根
    @param records 含 vector 字段的记录列表
    @return 写入条数
    """
    if not records:
        return 0
    dim = len(records[0].get("vector", []))
    table = get_table(workspace, vector_dim=dim or DEFAULT_VECTOR_DIM)
    table.add(records)
    return len(records)


def delete_chunks_for_file(workspace: Path, rel_path: str) -> None:
    """
    删除某文件的全部 chunk。

    @param workspace 工作区根
    @param rel_path 相对路径
    """
    db = connect_db(workspace)
    if TABLE_NAME not in db.table_names():
        return
    table = db.open_table(TABLE_NAME)
    escaped = rel_path.replace('"', '\\"')
    try:
        table.delete(f'rel_path = "{escaped}"')
    except Exception:
        pass


def delete_chunks_under_top_dir(workspace: Path, top_dir: str) -> None:
    """
    删除某顶层目录下全部 chunk（用于 skip_dirs 批量清理）。

    @param workspace 工作区根
    @param top_dir 工作区根下目录名，不含 /
    """
    name = top_dir.strip().strip("/")
    if not name:
        return
    db = connect_db(workspace)
    if TABLE_NAME not in db.table_names():
        return
    table = db.open_table(TABLE_NAME)
    escaped = name.replace('"', '\\"')
    try:
        table.delete(f'rel_path LIKE "{escaped}/%"')
    except Exception:
        pass


def delete_chunks_not_in_files(workspace: Path, keep_paths: set[str]) -> int:
    """
    删除已不在工作区 manifest 中的文件的 chunk。

    @param workspace 工作区根
    @param keep_paths 仍存在的 rel_path 集合
    @return 删除的文件数（近似）
    """
    db = connect_db(workspace)
    if TABLE_NAME not in db.table_names():
        return 0
    table = db.open_table(TABLE_NAME)
    try:
        df = table.to_pandas()
    except Exception:
        return 0
    if df.empty or "rel_path" not in df.columns:
        return 0
    stale = set(df["rel_path"].unique()) - keep_paths
    for rel in stale:
        delete_chunks_for_file(workspace, rel)
    return len(stale)


def search_vectors(
    workspace: Path,
    query_vector: list[float],
    *,
    top_k: int = DEFAULT_SEARCH_TOP_K,
    path_prefix: str = "",
) -> list[dict[str, Any]]:
    """
    向量最近邻检索。

    @param workspace 工作区根
    @param query_vector 查询向量
    @param top_k 返回条数
    @param path_prefix 限制 rel_path 前缀
    @return 命中记录列表（含 _distance）
    """
    db = connect_db(workspace)
    if TABLE_NAME not in db.table_names():
        return []
    table = db.open_table(TABLE_NAME)
    try:
        count = table.count_rows()
    except Exception:
        count = 0
    if count == 0:
        return []

    q = table.search(query_vector).limit(top_k)
    if path_prefix and path_prefix not in (".", ""):
        prefix = path_prefix.strip().rstrip("/")
        q = q.where(f"rel_path LIKE '{prefix}/%' OR rel_path = '{prefix}'")
    results = q.to_list()
    return results


def get_index_status(workspace: Path) -> IndexStatus:
    """
    返回索引状态。

    @param workspace 工作区根
    @return IndexStatus
    """
    root = index_root(workspace)
    meta = _load_meta(workspace)
    lance_path = str(root / "lance")
    db = connect_db(workspace)
    if TABLE_NAME not in db.table_names():
        return IndexStatus(
            exists=False,
            chunk_count=0,
            vector_dim=meta.get("vector_dim", DEFAULT_VECTOR_DIM),
            last_indexed_at=meta.get("last_indexed_at"),
            lance_path=lance_path,
        )
    table = db.open_table(TABLE_NAME)
    try:
        count = table.count_rows()
    except Exception:
        count = 0
    return IndexStatus(
        exists=True,
        chunk_count=count,
        vector_dim=meta.get("vector_dim", DEFAULT_VECTOR_DIM),
        last_indexed_at=meta.get("last_indexed_at"),
        lance_path=lance_path,
    )


def touch_index_meta(
    workspace: Path,
    *,
    vector_dim: int | None = None,
    sync_files: int | None = None,
    sync_complete: bool | None = None,
) -> None:
    """
    更新索引元信息。

    @param workspace 工作区根
    @param vector_dim 向量维度
    @param sync_files manifest 中文件数
    @param sync_complete 是否已完成一次无上限的全库遍历
    """
    meta = _load_meta(workspace)
    meta["last_indexed_at"] = datetime.now(timezone.utc).isoformat()
    if vector_dim is not None:
        meta["vector_dim"] = vector_dim
    if sync_files is not None:
        meta["sync_files"] = sync_files
    if sync_complete is not None:
        meta["sync_complete"] = sync_complete
    _save_meta(workspace, meta)


def record_index_suffixes(workspace: Path) -> None:
    """
    将当前 embedding.json 中的索引后缀写入 index_meta.json（便于排查与对齐）。

    @param workspace 工作区根
    """
    from llgraph.core.text_file_types import resolve_grep_suffixes, resolve_index_suffixes

    meta = _load_meta(workspace)
    meta["include_suffixes"] = list(resolve_index_suffixes(workspace))
    meta["grep_include_suffixes"] = list(resolve_grep_suffixes(workspace))
    _save_meta(workspace, meta)
