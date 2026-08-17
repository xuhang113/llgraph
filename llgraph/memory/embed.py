"""记忆文本向量化（复用 code_index embedder）。"""

from __future__ import annotations

import hashlib
from pathlib import Path

from llgraph.code_index.embedder import embed_texts


def content_hash(text: str) -> str:
    """内容 hash。"""
    normalized = " ".join((text or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def embed_memory_text(workspace: Path, text: str) -> list[float]:
    """
    单条记忆 embed。

    @param workspace 工作区根（embed 缓存路径）
    @param text 正文
    @return 向量
    """
    body = (text or "").strip()
    if not body:
        return []
    h = content_hash(body)
    vecs = embed_texts(workspace, [body], [h])
    if isinstance(vecs, tuple):
        vecs = vecs[0]
    return list(vecs[0]) if vecs else []
