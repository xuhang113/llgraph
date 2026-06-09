"""Embedding：本地 sentence-transformers（默认）或 Gateway /v1/embeddings + SQLite 缓存。"""

from __future__ import annotations

import json
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path

from llgraph.code_index.embedding_config import EmbeddingProfile, resolve_embedding_profile
from llgraph.code_index.local_embedder import embed_texts_local, get_local_embedding_dimension
from llgraph.code_index.paths import embed_cache_path, ensure_index_dirs


def clear_embed_cache(workspace: Path) -> bool:
    """
    删除 embedding SQLite 缓存。

    @param workspace 工作区根
    @return 是否删除了文件
    """
    path = embed_cache_path(workspace)
    if path.is_file():
        path.unlink()
        return True
    return False


def _cache_connect(workspace: Path) -> sqlite3.Connection:
    ensure_index_dirs(workspace)
    conn = sqlite3.connect(embed_cache_path(workspace))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embed_cache (
            content_hash TEXT PRIMARY KEY,
            vector_json TEXT NOT NULL,
            model TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def get_cached_vector(workspace: Path, content_hash: str, model: str) -> list[float] | None:
    """
    从缓存读取向量。

    @param workspace 工作区根
    @param content_hash 内容 hash
    @param model 缓存键（provider:model）
    @return 向量或 None
    """
    conn = _cache_connect(workspace)
    try:
        row = conn.execute(
            "SELECT vector_json FROM embed_cache WHERE content_hash = ? AND model = ?",
            (content_hash, model),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    return json.loads(row[0])


def put_cached_vector(
    workspace: Path,
    content_hash: str,
    model: str,
    vector: list[float],
) -> None:
    """写入缓存。"""
    put_cached_vectors_batch(workspace, [(content_hash, vector)], model)


def lookup_cached_vectors_batch(
    workspace: Path,
    content_hashes: list[str],
    model: str,
) -> dict[str, list[float]]:
    """
    批量读取 embed 缓存（单次连接 + IN 查询）。

    @param workspace 工作区根
    @param content_hashes 内容 hash 列表
    @param model 缓存键
    @return content_hash -> 向量
    """
    if not content_hashes:
        return {}

    unique_hashes = list(dict.fromkeys(content_hashes))
    found: dict[str, list[float]] = {}
    conn = _cache_connect(workspace)
    try:
        # SQLite 单条 SQL 绑定变量上限约 999
        chunk_size = 400
        for offset in range(0, len(unique_hashes), chunk_size):
            batch = unique_hashes[offset : offset + chunk_size]
            placeholders = ",".join("?" * len(batch))
            sql = (
                f"SELECT content_hash, vector_json FROM embed_cache "
                f"WHERE model = ? AND content_hash IN ({placeholders})"
            )
            rows = conn.execute(sql, [model, *batch]).fetchall()
            for content_hash, vector_json in rows:
                found[content_hash] = json.loads(vector_json)
    finally:
        conn.close()
    return found


def put_cached_vectors_batch(
    workspace: Path,
    items: list[tuple[str, list[float]]],
    model: str,
) -> None:
    """
    批量写入 embed 缓存。

    @param workspace 工作区根
    @param items (content_hash, vector) 列表
    @param model 缓存键
    """
    if not items:
        return
    now = time.time()
    rows = [(h, json.dumps(vec), model, now) for h, vec in items]
    conn = _cache_connect(workspace)
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO embed_cache (content_hash, vector_json, model, created_at)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def get_embedding_dimension(workspace: Path) -> int:
    """
    当前配置下的向量维度。

    @param workspace 工作区根
    @return 维度
    """
    profile = resolve_embedding_profile(workspace)
    if profile.provider == "local":
        return get_local_embedding_dimension(profile)
    if profile.dimension is not None:
        return profile.dimension
    return 1536


def format_embedding_status(workspace: Path, *, probe_dim: bool = False) -> str:
    """
    用于 status 展示的 embedding 摘要。

    @param workspace 工作区根
    @param probe_dim 为 True 时加载本地模型探测维度（索引开始时使用）
    @return 单行描述
    """
    try:
        profile = resolve_embedding_profile(workspace)
    except RuntimeError as exc:
        return f"配置错误 ({exc})"

    if profile.provider == "local":
        if profile.dimension is not None:
            dim_text = str(profile.dimension)
        elif probe_dim:
            dim_text = str(get_local_embedding_dimension(profile))
        else:
            dim_text = "auto"
        offline = "offline" if profile.local_files_only else "online-ok"
        return f"local {profile.model} (dim={dim_text}, device={profile.device}, {offline})"
    dim = profile.dimension or "?"
    return f"remote {profile.model} @ {profile.base_url} (dim={dim})"


def _embed_batch_remote(texts: list[str], profile: EmbeddingProfile) -> list[list[float]]:
    """
    调用 Gateway /v1/embeddings。

    @param texts 文本列表
    @param profile embedding 配置
    @return 向量列表
    """
    if not profile.base_url or not profile.api_key:
        raise RuntimeError("remote embedding 缺少 base_url 或 api_key")

    url = f"{profile.base_url.rstrip('/')}/v1/embeddings"
    body = json.dumps({"model": profile.model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {profile.api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Embedding API 失败 HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Embedding API 连接失败: {exc}") from exc

    data = payload.get("data", [])
    if not data:
        raise RuntimeError(f"Embedding API 返回无 data: {payload!r}")
    sorted_items = sorted(data, key=lambda x: x.get("index", 0))
    return [item["embedding"] for item in sorted_items]


def _embed_batch(texts: list[str], profile: EmbeddingProfile) -> list[list[float]]:
    """
    按 provider 批量向量化。

    @param texts 文本列表
    @param profile embedding 配置
    @return 向量列表
    """
    if profile.provider == "local":
        return embed_texts_local(texts, profile)
    return _embed_batch_remote(texts, profile)


def embed_texts(
    workspace: Path,
    texts: list[str],
    content_hashes: list[str],
    *,
    batch_size: int | None = None,
    use_embed_cache: bool | None = None,
    return_stats: bool = False,
) -> list[list[float]] | tuple[list[list[float]], dict[str, int]]:
    """
    批量 embed；可选 SQLite 缓存（默认关，见 index.use_embed_cache）。

    @param workspace 工作区根
    @param texts 与 content_hashes 等长
    @param content_hashes 内容 hash
    @param batch_size 覆盖配置中的批大小
    @param use_embed_cache 是否查/写 embed_cache.db；None 时读 embedding.json
    @return 向量列表
    """
    if len(texts) != len(content_hashes):
        raise ValueError("texts 与 content_hashes 长度不一致")

    if not texts:
        stats = {"total": 0, "cache_hits": 0, "computed": 0, "cache_enabled": False}
        if return_stats:
            return [], stats  # type: ignore[return-value]
        return []

    profile = resolve_embedding_profile(workspace)
    effective_batch = batch_size if batch_size is not None else profile.batch_size

    if use_embed_cache is None:
        from llgraph.code_index.index_settings import resolve_index_settings

        use_embed_cache = resolve_index_settings(workspace).use_embed_cache

    if not use_embed_cache:
        results: list[list[float]] = []
        for offset in range(0, len(texts), effective_batch):
            batch_text = texts[offset : offset + effective_batch]
            results.extend(_embed_batch(batch_text, profile))
        stats = {
            "total": len(texts),
            "cache_hits": 0,
            "computed": len(texts),
            "cache_enabled": False,
        }
        if return_stats:
            return results, stats  # type: ignore[return-value]
        return results

    cache_key = profile.cache_model_key
    results: list[list[float] | None] = [None] * len(texts)
    pending_indices: list[int] = []
    pending_texts: list[str] = []

    cache_hits = lookup_cached_vectors_batch(workspace, content_hashes, cache_key)
    for idx, chash in enumerate(content_hashes):
        cached = cache_hits.get(chash)
        if cached is not None:
            results[idx] = cached
        else:
            pending_indices.append(idx)
            pending_texts.append(texts[idx])

    cache_writes: list[tuple[str, list[float]]] = []
    for offset in range(0, len(pending_texts), effective_batch):
        batch_idx = pending_indices[offset : offset + effective_batch]
        batch_text = pending_texts[offset : offset + effective_batch]
        vectors = _embed_batch(batch_text, profile)
        for i, vec in zip(batch_idx, vectors):
            results[i] = vec
            cache_writes.append((content_hashes[i], vec))

    if cache_writes:
        put_cached_vectors_batch(workspace, cache_writes, cache_key)

    if any(r is None for r in results):
        raise RuntimeError("embedding 结果不完整")

    stats = {
        "total": len(texts),
        "cache_hits": len(texts) - len(pending_texts),
        "computed": len(pending_texts),
        "cache_enabled": True,
    }
    if return_stats:
        return results, stats  # type: ignore[return-value]
    return results  # type: list[list[float]]


def embed_query(workspace: Path, query: str) -> list[float]:
    """
    单条查询 embed（查询不走内容 hash 缓存）。

    @param workspace 工作区根
    @param query 查询文本
    @return 向量
    """
    profile = resolve_embedding_profile(workspace)
    vectors = _embed_batch([query], profile)
    return vectors[0]
