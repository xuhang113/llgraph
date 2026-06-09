"""Embedding 配置：本地 sentence-transformers / 远程 Gateway，支持配置文件与环境变量。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from llgraph.config.config import load_llgraph_env

EmbeddingProvider = Literal["local", "remote"]

DEFAULT_LOCAL_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_REMOTE_MODEL = "text-embedding-3-small"
DEFAULT_REMOTE_DIMENSION = 1536
EMBEDDING_CONFIG_FILENAME = "embedding.json"
USER_CONFIG_PATH = Path.home() / ".config" / "llgraph" / EMBEDDING_CONFIG_FILENAME


@dataclass(frozen=True)
class EmbeddingProfile:
    """解析后的 embedding 配置。"""

    provider: EmbeddingProvider
    model: str
    dimension: int | None
    batch_size: int
    device: str
    normalize: bool
    local_files_only: bool
    base_url: str | None
    api_key: str | None

    @property
    def cache_model_key(self) -> str:
        """SQLite 缓存用的模型标识（区分 local/remote）。"""
        return f"{self.provider}:{self.model}"


def _workspace_embedding_path(workspace: Path) -> Path:
    return workspace.expanduser().resolve() / ".llgraph" / EMBEDDING_CONFIG_FILENAME


def _load_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _default_config() -> dict[str, Any]:
    return {
        "provider": "local",
        "local": {
            "model": DEFAULT_LOCAL_MODEL,
            "device": "auto",
            "batch_size": 32,
            "normalize": True,
            "local_files_only": False,
        },
        "remote": {
            "model": DEFAULT_REMOTE_MODEL,
            "dimension": DEFAULT_REMOTE_DIMENSION,
            "batch_size": 32,
            "base_url": "",
            "api_key": "",
        },
        "index": {
            "max_files": 0,
            "progressive": True,
            "manifest_flush_every": 50,
            "progress_log_every": 500,
            "show_progress": True,
            "prepare_workers": 4,
            "embed_accumulate_chunks": 128,
            "use_embed_cache": False,
            "include_suffixes": [
                ".py", ".java", ".kt", ".kts", ".go", ".rs",
                ".js", ".ts", ".tsx", ".jsx", ".md", ".mdc", ".txt",
                ".yaml", ".yml", ".json", ".xml", ".properties", ".sql",
                ".sh", ".zsh", ".toml", ".ini", ".cfg", ".html", ".css",
                ".vue", ".gradle", ".pom", ".csv", ".proto",
            ],
            "grep": {
                "include_suffixes": [
                    ".py", ".java", ".kt", ".kts", ".go", ".rs",
                    ".js", ".ts", ".tsx", ".jsx", ".md", ".mdc", ".txt",
                    ".yaml", ".yml", ".json", ".xml", ".properties", ".sql",
                    ".sh", ".zsh", ".toml", ".ini", ".cfg", ".html", ".css",
                    ".vue", ".gradle", ".pom", ".csv", ".proto",
                ]
            },
        },
    }


def _apply_env(cfg: dict[str, Any]) -> dict[str, Any]:
    load_llgraph_env()
    provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    if provider in ("local", "remote"):
        cfg["provider"] = provider

    local_model = os.getenv("EMBEDDING_LOCAL_MODEL", "").strip()
    if local_model:
        cfg.setdefault("local", {})["model"] = local_model

    device = os.getenv("EMBEDDING_DEVICE", "").strip()
    if device:
        cfg.setdefault("local", {})["device"] = device

    model = os.getenv("EMBEDDING_MODEL", "").strip()
    if model:
        prov = cfg.get("provider", "local")
        if prov == "remote":
            cfg.setdefault("remote", {})["model"] = model
        else:
            cfg.setdefault("local", {})["model"] = model

    dim_raw = os.getenv("EMBEDDING_DIMENSION", "").strip()
    if dim_raw.isdigit():
        cfg.setdefault("remote", {})["dimension"] = int(dim_raw)

    batch_raw = os.getenv("EMBEDDING_BATCH_SIZE", "").strip()
    if batch_raw.isdigit():
        size = int(batch_raw)
        cfg.setdefault("local", {})["batch_size"] = size
        cfg.setdefault("remote", {})["batch_size"] = size

    base_url = os.getenv("EMBEDDING_BASE_URL", "").strip()
    if base_url:
        cfg.setdefault("remote", {})["base_url"] = base_url

    api_key = os.getenv("EMBEDDING_API_KEY", "").strip()
    if api_key:
        cfg.setdefault("remote", {})["api_key"] = api_key

    offline = os.getenv("HF_HUB_OFFLINE", "").strip()
    if offline in ("1", "true", "yes"):
        cfg.setdefault("local", {})["local_files_only"] = True

    return cfg


def load_embedding_config(workspace: Path | None = None) -> dict[str, Any]:
    """
    加载 embedding 配置（默认 + 用户 + 工作区 + 环境变量）。

    @param workspace 工作区根，可为 None
    @return 合并后的配置 dict
    """
    cfg = _default_config()
    cfg = _deep_merge(cfg, _load_json_file(USER_CONFIG_PATH))
    if workspace is not None:
        cfg = _deep_merge(cfg, _load_json_file(_workspace_embedding_path(workspace)))
    return _apply_env(cfg)


def resolve_embedding_profile(workspace: Path) -> EmbeddingProfile:
    """
    解析为 EmbeddingProfile；remote 时校验网关凭据。

    @param workspace 工作区根
    @return EmbeddingProfile
    """
    cfg = load_embedding_config(workspace)
    provider_raw = str(cfg.get("provider", "local")).strip().lower()
    provider: EmbeddingProvider = "remote" if provider_raw == "remote" else "local"

    if provider == "local":
        local = cfg.get("local") if isinstance(cfg.get("local"), dict) else {}
        model = str(local.get("model", DEFAULT_LOCAL_MODEL)).strip() or DEFAULT_LOCAL_MODEL
        device = str(local.get("device", "auto")).strip() or "auto"
        batch_size = int(local.get("batch_size", 32))
        normalize = bool(local.get("normalize", True))
        local_only = local.get("local_files_only", False)
        if isinstance(local_only, str):
            local_only = local_only.strip().lower() in ("1", "true", "yes")
        else:
            local_only = bool(local_only)
        dim = local.get("dimension")
        dimension = int(dim) if isinstance(dim, int) or (isinstance(dim, str) and str(dim).isdigit()) else None
        return EmbeddingProfile(
            provider="local",
            model=model,
            dimension=dimension,
            batch_size=max(1, batch_size),
            device=device,
            normalize=normalize,
            local_files_only=local_only,
            base_url=None,
            api_key=None,
        )

    remote = cfg.get("remote") if isinstance(cfg.get("remote"), dict) else {}
    load_llgraph_env()
    base_url = (
        str(remote.get("base_url", "")).strip()
        or os.getenv("EMBEDDING_BASE_URL", "").strip()
        or os.getenv("LLGRAPH_API_BASE_URL", "").strip()
    )
    api_key = (
        str(remote.get("api_key", "")).strip()
        or os.getenv("EMBEDDING_API_KEY", "").strip()
        or os.getenv("LLGRAPH_API_KEY", "").strip()
    )
    model = (
        str(remote.get("model", "")).strip()
        or os.getenv("EMBEDDING_MODEL", "").strip()
        or DEFAULT_REMOTE_MODEL
    )
    dim_raw = remote.get("dimension", DEFAULT_REMOTE_DIMENSION)
    try:
        dimension = int(dim_raw)
    except (TypeError, ValueError):
        dimension = DEFAULT_REMOTE_DIMENSION
    batch_size = int(remote.get("batch_size", 32))

    missing = []
    if not base_url:
        missing.append("remote.base_url / EMBEDDING_BASE_URL / LLGRAPH_API_BASE_URL")
    if not api_key:
        missing.append("remote.api_key / EMBEDDING_API_KEY / LLGRAPH_API_KEY")
    if missing:
        raise RuntimeError(
            f"Embedding provider=remote 但缺少: {', '.join(missing)}。"
            "请在 .llgraph/embedding.json 或 ~/.config/llgraph/llgraph.env 中配置。"
        )

    return EmbeddingProfile(
        provider="remote",
        model=model,
        dimension=dimension,
        batch_size=max(1, batch_size),
        device="cpu",
        normalize=False,
        local_files_only=False,
        base_url=base_url.rstrip("/"),
        api_key=api_key,
    )


def embedding_config_paths(workspace: Path) -> list[Path]:
    """返回可能生效的配置文件路径（用于 status 展示）。"""
    paths = [USER_CONFIG_PATH, _workspace_embedding_path(workspace)]
    return paths
