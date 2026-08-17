"""长期记忆存储路径与常量。"""

from __future__ import annotations

import json
from pathlib import Path

from llgraph.code_index.paths import CHUNK_TARGET_CHARS, DEFAULT_VECTOR_DIM
from llgraph.core.agent_config import user_llgraph_home
from llgraph.session.user_storage import workspace_context_slug, workspace_storage_key

MEMORY_DIR_NAME = "memory"
LANCE_SUBDIR = "lance"
TABLE_NAME = "agent_memories"
META_FILENAME = "memory_meta.json"
CONSOLIDATE_REPORT = "consolidate_report.jsonl"
LOCK_FILENAME = ".memory_consolidate.lock"

DEFAULT_CONTENT_MAX_CHARS = CHUNK_TARGET_CHARS
DEFAULT_INJECT_EXCERPT_CHARS = 800
DELETE_BATCH_SIZE = 40

KIND_PREF = "pref"
KIND_FACT = "fact"
KIND_PROC = "proc"
ACTIVE_KINDS = (KIND_PREF, KIND_FACT, KIND_PROC)

STATUS_ACTIVE = "active"
STATUS_SUPERSEDED = "superseded"
STATUS_REJECTED = "rejected"


def resolve_memory_user_id(workspace: Path, configured: str | None = None) -> str:
    """
    解析记忆 user_id。

    @param workspace 工作区根
    @param configured agent.json context.memory.user_id
    @return 用户标识
    """
    import os

    if configured and str(configured).strip():
        return str(configured).strip()
    env = os.environ.get("LLGRAPH_MEMORY_USER_ID", "").strip()
    if env:
        return env
    return os.environ.get("USER", "").strip() or "default"


def memory_root(user_id: str, workspace_key: str) -> Path:
    """~/.llgraph/memory/<user_id>/<workspace_key>/（或 $LLGRAPH_HOME/memory/...）"""
    uid = (user_id or "default").strip() or "default"
    key = (workspace_key or "").strip()
    return user_llgraph_home() / MEMORY_DIR_NAME / uid / key


def memory_lance_uri(user_id: str, workspace_key: str) -> str:
    """LanceDB URI。"""
    return str(memory_root(user_id, workspace_key) / LANCE_SUBDIR)


def memory_meta_path(user_id: str, workspace_key: str) -> Path:
    """memory_meta.json 路径。"""
    return memory_root(user_id, workspace_key) / META_FILENAME


def ensure_memory_dirs(user_id: str, workspace_key: str) -> Path:
    """
    创建记忆目录。

    @param user_id 用户 ID
    @param workspace_key 工作区键
    @return 记忆根路径
    """
    root = memory_root(user_id, workspace_key)
    root.mkdir(parents=True, exist_ok=True)
    (root / LANCE_SUBDIR).mkdir(parents=True, exist_ok=True)
    return root


def workspace_keys_for_user(user_id: str) -> list[str]:
    """
    枚举用户下已有 workspace_key 目录。

    @param user_id 用户 ID
    @return workspace_key 列表
    """
    uid = (user_id or "default").strip() or "default"
    base = user_llgraph_home() / MEMORY_DIR_NAME / uid
    if not base.is_dir():
        return []
    out: list[str] = []
    for child in base.iterdir():
        if child.is_dir() and (child / LANCE_SUBDIR).is_dir():
            out.append(child.name)
    return sorted(out)


def load_memory_meta(user_id: str, workspace_key: str) -> dict:
    """读取 memory_meta.json。"""
    path = memory_meta_path(user_id, workspace_key)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_memory_meta(user_id: str, workspace_key: str, patch: dict) -> None:
    """合并写入 memory_meta.json。"""
    path = memory_meta_path(user_id, workspace_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_memory_meta(user_id, workspace_key)
    existing.update(patch)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def workspace_identity(workspace: Path, *, user_id: str | None = None) -> tuple[str, str, str]:
    """
    返回 (user_id, workspace_key, workspace_slug)。

    @param workspace 工作区根
    @param user_id 可选覆盖
    """
    from llgraph.memory.settings import resolve_memory_settings

    settings = resolve_memory_settings(workspace)
    uid = resolve_memory_user_id(workspace, user_id or settings.user_id)
    key = workspace_storage_key(workspace)
    slug = workspace_context_slug(workspace)
    return uid, key, slug
