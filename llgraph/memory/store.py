"""LanceDB agent_memories 表 CRUD。"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from llgraph.code_index.paths import DEFAULT_VECTOR_DIM
from llgraph.memory.paths import (
    STATUS_ACTIVE,
    TABLE_NAME,
    ensure_memory_dirs,
    memory_lance_uri,
    memory_store_is_definitely_empty,
    save_memory_meta,
)


def _require_lancedb():
    try:
        import lancedb
    except ImportError as exc:
        raise RuntimeError(
            "未安装 lancedb。请执行: pip install 'llgraph[index]' 或 pip install lancedb"
        ) from exc
    return lancedb


def _memory_record(
    *,
    memory_id: str,
    user_id: str,
    workspace_key: str,
    workspace_slug: str,
    kind: str,
    content: str,
    content_hash: str,
    vector: list[float],
    confidence: float,
    source: str,
    status: str,
    supersedes_id: str,
    created_at: str,
    updated_at: str,
    last_hit_at: str,
    hit_count: int,
    weight_boost: float,
    tags: str,
) -> dict[str, Any]:
    return {
        "memory_id": memory_id,
        "user_id": user_id,
        "workspace_key": workspace_key,
        "workspace_slug": workspace_slug,
        "kind": kind,
        "content": content,
        "content_hash": content_hash,
        "vector": vector,
        "confidence": float(confidence),
        "source": source,
        "status": status,
        "supersedes_id": supersedes_id or "",
        "created_at": created_at,
        "updated_at": updated_at,
        "last_hit_at": last_hit_at or "",
        "hit_count": int(hit_count),
        "weight_boost": float(weight_boost),
        "tags": tags or "[]",
    }


def connect_memory_db(user_id: str, workspace_key: str):
    """连接记忆 LanceDB。"""
    ensure_memory_dirs(user_id, workspace_key)
    lancedb = _require_lancedb()
    return lancedb.connect(memory_lance_uri(user_id, workspace_key))


def _table_names(db) -> list[str]:
    """兼容 lancedb table_names / list_tables。"""
    if hasattr(db, "list_tables"):
        resp = db.list_tables()
        if hasattr(resp, "tables"):
            return list(resp.tables)
        return list(resp)
    return list(db.table_names())


def get_memory_table(
    user_id: str,
    workspace_key: str,
    *,
    vector_dim: int = DEFAULT_VECTOR_DIM,
):
    """打开或创建 agent_memories 表。"""
    db = connect_memory_db(user_id, workspace_key)
    if TABLE_NAME in _table_names(db):
        return db.open_table(TABLE_NAME)
    placeholder = _memory_record(
        memory_id="__placeholder__",
        user_id=user_id,
        workspace_key=workspace_key,
        workspace_slug="",
        kind="pref",
        content="",
        content_hash="",
        vector=[0.0] * vector_dim,
        confidence=0.0,
        source="",
        status=STATUS_ACTIVE,
        supersedes_id="",
        created_at="",
        updated_at="",
        last_hit_at="",
        hit_count=0,
        weight_boost=0.0,
        tags="[]",
    )
    table = db.create_table(TABLE_NAME, data=[placeholder])
    table.delete('memory_id = "__placeholder__"')
    save_memory_meta(
        user_id,
        workspace_key,
        {"table": TABLE_NAME, "vector_dim": vector_dim},
    )
    return table


def _table_row_dicts(table, *, limit: int = 5000) -> list[dict[str, Any]]:
    """读取表行（不依赖 pylance/pandas）。"""
    try:
        total = table.count_rows()
        if total <= 0:
            return []
        batch = min(limit, total)
        return table.head(batch).to_pylist()
    except Exception:
        return []


def list_memory_rows(
    user_id: str,
    workspace_key: str,
    *,
    status: str | None = STATUS_ACTIVE,
    kinds: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """
    列出记忆行（pandas 过滤）。

    @param user_id 用户 ID
    @param workspace_key 工作区键
    @param status 状态过滤；None 表示全部
    @param kinds kind 白名单
    @return 行 dict 列表
    """
    if memory_store_is_definitely_empty(user_id, workspace_key):
        return []
    db = connect_memory_db(user_id, workspace_key)
    if TABLE_NAME not in _table_names(db):
        return []
    table = db.open_table(TABLE_NAME)
    rows = _table_row_dicts(table)
    if not rows:
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("user_id") != user_id or row.get("workspace_key") != workspace_key:
            continue
        if status and row.get("status") != status:
            continue
        if kinds and row.get("kind") not in kinds:
            continue
        out.append(row)
    return out


def search_memory_vectors(
    user_id: str,
    workspace_key: str,
    query_vector: list[float],
    *,
    top_k: int = 20,
    kinds: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """
    向量检索 active 记忆。

    @param user_id 用户 ID
    @param workspace_key 工作区键
    @param query_vector 查询向量
    @param top_k 条数
    @param kinds kind 白名单
    @return 命中行
    """
    if memory_store_is_definitely_empty(user_id, workspace_key):
        return []
    db = connect_memory_db(user_id, workspace_key)
    if TABLE_NAME not in _table_names(db):
        return []
    table = db.open_table(TABLE_NAME)
    try:
        if table.count_rows() == 0:
            return []
    except Exception:
        return []
    uid = user_id.replace("'", "''")
    wk = workspace_key.replace("'", "''")
    where = f"user_id = '{uid}' AND workspace_key = '{wk}' AND status = 'active'"
    if kinds:
        kinds_sql = ", ".join(f"'{k}'" for k in kinds)
        where += f" AND kind IN ({kinds_sql})"
    try:
        results = table.search(query_vector).where(where).limit(top_k).to_list()
    except Exception:
        results = table.search(query_vector).limit(top_k).to_list()
        results = [
            r
            for r in results
            if r.get("user_id") == user_id
            and r.get("workspace_key") == workspace_key
            and r.get("status") == STATUS_ACTIVE
            and (not kinds or r.get("kind") in kinds)
        ]
    return results


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """两向量余弦相似度。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def utc_now_iso() -> str:
    """当前 UTC ISO 时间。"""
    return datetime.now(timezone.utc).isoformat()


def count_active_memories(user_id: str, workspace_key: str) -> int:
    """active 条数。"""
    return len(list_memory_rows(user_id, workspace_key, status=STATUS_ACTIVE))
