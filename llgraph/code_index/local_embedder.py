"""本地 Embedding：sentence-transformers。"""

from __future__ import annotations

import os
from typing import Any

from llgraph.code_index.embedding_config import EmbeddingProfile

_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_DIMENSION_CACHE: dict[str, int] = {}


def _resolve_device(device: str) -> str:
    """
    解析运行设备。

    @param device auto|cpu|mps|cuda
    @return sentence-transformers 可用 device 字符串
    """
    normalized = device.strip().lower()
    if normalized and normalized != "auto":
        return normalized

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_model(profile: EmbeddingProfile):
    """
    懒加载 SentenceTransformer 模型。

    @param profile embedding 配置
    @return SentenceTransformer 实例
    """
    device = _resolve_device(profile.device)
    cache_key = (profile.model, device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # 避免「Loading weights」等进度条刷进交互对话区
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "未安装 sentence-transformers。请执行: pip install -e '.[index]'"
        ) from exc

    model_id = profile.model
    load_kwargs: dict[str, Any] = {"device": device}

    if profile.local_files_only:
        # 显式离线：仅用本地缓存，缺失则报错（可配合 HF_HUB_OFFLINE=1）
        model = SentenceTransformer(model_id, local_files_only=True, **load_kwargs)
        _MODEL_CACHE[cache_key] = model
        return model

    # 默认：先按模型名走本地缓存（避免 Hub HEAD 超时），缓存不全再联网下载
    try:
        model = SentenceTransformer(model_id, local_files_only=True, **load_kwargs)
    except Exception:
        model = SentenceTransformer(model_id, **load_kwargs)

    _MODEL_CACHE[cache_key] = model
    return model


def get_local_embedding_dimension(profile: EmbeddingProfile) -> int:
    """
    获取本地模型向量维度（带缓存）。

    @param profile embedding 配置
    @return 维度
    """
    if profile.dimension is not None and profile.dimension > 0:
        return profile.dimension
    if profile.model in _DIMENSION_CACHE:
        return _DIMENSION_CACHE[profile.model]

    model = _get_model(profile)
    if hasattr(model, "get_embedding_dimension"):
        dim = int(model.get_embedding_dimension())
    else:
        dim = int(model.get_sentence_embedding_dimension())
    _DIMENSION_CACHE[profile.model] = dim
    return dim


def embed_texts_local(texts: list[str], profile: EmbeddingProfile) -> list[list[float]]:
    """
    本地批量向量化。

    @param texts 文本列表
    @param profile embedding 配置
    @return 向量列表
    """
    if not texts:
        return []

    model = _get_model(profile)
    encode_kwargs: dict[str, Any] = {
        "batch_size": profile.batch_size,
        "show_progress_bar": False,
        "convert_to_numpy": True,
    }
    if profile.normalize:
        encode_kwargs["normalize_embeddings"] = True

    vectors = model.encode(texts, **encode_kwargs)
    return [vec.tolist() for vec in vectors]
