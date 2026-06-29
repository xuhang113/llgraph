"""用户级存储路径（~/.llgraph/context，按工作区目录隔离）。"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

from llgraph.core.agent_config import USER_LLGRAPH_HOME

RULES_DIR_NAME = "rules"
SKILLS_DIR_NAME = "skills"
CONTEXT_DIR_NAME = "context"
SESSIONS_DIR_NAME = "sessions"
MESSAGES_FILENAME = "messages.jsonl"
EDITS_FILENAME = "edits.jsonl"
SNAPSHOTS_DIR_NAME = "snapshots"
ATTACHMENTS_DIR_NAME = "attachments"
LEGACY_WORKSPACE_SESSIONS_DIR = ".llgraph/sessions"
# agent.json 中 edits.sessions_dir 为此值时，改用用户目录（与 messages.jsonl 同目录）
LEGACY_SESSIONS_DIR_CONFIG = ".llgraph/sessions"


def user_rules_dir() -> Path:
    """
    个人规则目录。

    @return ~/.llgraph/rules/
    """
    return USER_LLGRAPH_HOME / RULES_DIR_NAME


def user_skills_dir() -> Path:
    """
    个人技能目录。

    @return ~/.llgraph/skills/
    """
    return USER_LLGRAPH_HOME / SKILLS_DIR_NAME


def user_context_root() -> Path:
    """
    用户级上下文根目录。

    @return ~/.llgraph/context/
    """
    return USER_LLGRAPH_HOME / CONTEXT_DIR_NAME


def is_ephemeral_workspace_path(resolved: str) -> bool:
    """pytest tmp_path、系统临时目录等不应出现在 Web 最近工作区。"""
    lowered = resolved.lower()
    return (
        "/var/folders/" in lowered
        or "/private/var/folders/" in lowered
        or lowered.startswith("/tmp/")
        or lowered.startswith("/private/tmp/")
        or "/temporaryitems/" in lowered
    )


def workspace_storage_key(workspace: Path) -> str:
    """
    工作区稳定标识（同一路径始终相同）。

    @param workspace 工作区根
    @return 16 位 hex
    """
    normalized = str(workspace.expanduser().resolve())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def workspace_context_slug(workspace: Path) -> str:
    """
    可读的工作区目录名（用于 ~/.llgraph/context/<slug>/）。

    @param workspace 工作区根
    @return 如 my-monorepo-fc64a269
    """
    root = workspace.expanduser().resolve()
    base = re.sub(r"[^\w\-.]+", "-", root.name).strip("-") or "workspace"
    return f"{base}-{workspace_storage_key(workspace)[:8]}"


def workspace_context_dir(workspace: Path) -> Path:
    """
    当前工作区在 ~/.llgraph/context 下的目录。

    @param workspace 工作区根
    @return ~/.llgraph/context/<slug>/
    """
    path = user_context_root() / workspace_context_slug(workspace)
    _ensure_workspace_marker(path, workspace)
    return path


def user_sessions_root(workspace: Path) -> Path:
    """
    会话目录根。

    @param workspace 工作区根
    @return ~/.llgraph/context/<slug>/sessions/
    """
    return workspace_context_dir(workspace) / SESSIONS_DIR_NAME


def session_thread_dir(workspace: Path, thread_id: str) -> Path:
    """
    单会话目录。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @return .../sessions/<thread_id>/
    """
    safe_id = thread_id.replace("/", "_").strip() or "default"
    return user_sessions_root(workspace) / safe_id


def session_messages_path(workspace: Path, thread_id: str) -> Path:
    """
    对话正文 jsonl。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @return .../sessions/<thread_id>/messages.jsonl
    """
    return session_thread_dir(workspace, thread_id) / MESSAGES_FILENAME


def session_edits_path(workspace: Path, thread_id: str) -> Path:
    """
    编辑账本 edits.jsonl。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @return .../sessions/<thread_id>/edits.jsonl
    """
    return session_thread_dir(workspace, thread_id) / EDITS_FILENAME


def session_snapshots_dir(workspace: Path, thread_id: str) -> Path:
    """
    首次编辑前快照目录。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @return .../sessions/<thread_id>/snapshots/
    """
    return session_thread_dir(workspace, thread_id) / SNAPSHOTS_DIR_NAME


def session_attachments_dir(workspace: Path, thread_id: str) -> Path:
    """
    会话图片附件目录。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @return .../sessions/<thread_id>/attachments/
    """
    return session_thread_dir(workspace, thread_id) / ATTACHMENTS_DIR_NAME


def legacy_workspace_session_dir(workspace: Path, thread_id: str) -> Path:
    """
    旧版工作区内会话目录（待迁移）。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @return <工作区>/.llgraph/sessions/<thread_id>/
    """
    safe_id = thread_id.replace("/", "_").strip() or "default"
    return workspace.expanduser().resolve() / LEGACY_WORKSPACE_SESSIONS_DIR / safe_id


def resolve_session_storage_dir(
    workspace: Path,
    thread_id: str,
    *,
    sessions_dir_config: str | None = None,
) -> Path:
    """
    会话落盘目录（默认与 messages/manifest 同级的用户目录）。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @param sessions_dir_config agent.json edits.sessions_dir；仅绝对路径可覆盖
    @return 会话目录绝对路径
    """
    safe_id = thread_id.replace("/", "_").strip() or "default"
    raw = (sessions_dir_config or "").strip()
    if raw and raw != LEGACY_SESSIONS_DIR_CONFIG:
        base = Path(raw)
        if base.is_absolute():
            return base / safe_id
    return session_thread_dir(workspace, thread_id)


def migrate_legacy_workspace_session_dir(
    workspace: Path,
    thread_id: str,
    target_dir: Path,
) -> bool:
    """
    将工作区 .llgraph/sessions/<id>/ 下文件迁到用户会话目录（一次性）。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @param target_dir 目标目录（用户 sessions/<id>/）
    @return 是否发生过迁移
    """
    legacy = legacy_workspace_session_dir(workspace, thread_id)
    if not legacy.is_dir():
        return False
    target_dir.mkdir(parents=True, exist_ok=True)
    migrated = False
    for name in (EDITS_FILENAME, "meta.json"):
        src = legacy / name
        dst = target_dir / name
        if src.is_file() and not dst.is_file():
            try:
                shutil.move(str(src), str(dst))
                migrated = True
            except OSError:
                pass
    snap_src = legacy / SNAPSHOTS_DIR_NAME
    snap_dst = target_dir / SNAPSHOTS_DIR_NAME
    if snap_src.is_dir():
        try:
            if not snap_dst.exists():
                shutil.move(str(snap_src), str(snap_dst))
                migrated = True
            else:
                for item in snap_src.iterdir():
                    dest_item = snap_dst / item.name
                    if not dest_item.exists():
                        shutil.move(str(item), str(dest_item))
                        migrated = True
        except OSError:
            pass
    try:
        if legacy.is_dir() and not any(legacy.iterdir()):
            legacy.rmdir()
    except OSError:
        pass
    legacy_root = legacy.parent
    try:
        if legacy_root.is_dir() and not any(legacy_root.iterdir()):
            legacy_root.rmdir()
    except OSError:
        pass
    return migrated


def _ensure_workspace_marker(context_dir: Path, workspace: Path) -> None:
    """写入 workspace.json 便于人工辨认目录对应哪个项目。"""
    marker = context_dir / "workspace.json"
    if marker.is_file():
        return
    try:
        resolved = str(workspace.expanduser().resolve())
        payload: dict[str, str | bool] = {"path": resolved}
        if is_ephemeral_workspace_path(resolved):
            payload["hidden_from_recent"] = True
        context_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def cleanup_obsolete_session_storage(workspace: Path) -> None:
    """
    删除已废弃的会话库文件（若存在 checkpoints.sqlite 等试验残留）。

    @param workspace 工作区根
    """
    root = workspace.expanduser().resolve()
    to_delete: list[Path] = [root / ".llgraph" / "context" / "checkpoints.sqlite"]
    ctx_root = user_context_root()
    if ctx_root.is_dir():
        for path in ctx_root.rglob("checkpoints.sqlite*"):
            to_delete.append(path)
    seen: set[str] = set()
    for path in to_delete:
        key = str(path)
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        try:
            path.unlink()
        except OSError:
            pass

    # 移除仅用于旧版 hash 分桶的空目录
    legacy_bucket = ctx_root / "workspaces"
    if legacy_bucket.is_dir():
        try:
            shutil.rmtree(legacy_bucket, ignore_errors=True)
        except OSError:
            pass


def format_storage_location_hint(workspace: Path) -> str:
    """
    存储位置说明（banner / /config 用）。

    @param workspace 工作区根
    @return 一行说明
    """
    ctx = workspace_context_dir(workspace)
    return (
        f"会话数据: {ctx}/sessions/<thread_id>/"
        f"（messages.jsonl、edits.jsonl、manifest 等）"
    )
