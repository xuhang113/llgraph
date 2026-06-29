"""工作区发现与注册（Gateway 本地实现）。"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from llgraph.config.workspace_config import is_packaged_example_workspace
from llgraph.gateway.types import WorkspaceRecord
from llgraph.session.user_storage import (
    is_ephemeral_workspace_path,
    user_context_root,
    workspace_context_slug,
)


def resolve_context_root() -> Path:
    """
    解析 llgraph 用户 context 根目录。

    @return ~/.llgraph/context 或 LLGRAPH_HOME/context
    """
    home = os.environ.get("LLGRAPH_HOME", "").strip()
    if home:
        return Path(home).expanduser().resolve() / "context"
    return user_context_root()


def _mtime_iso(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except OSError:
        return None


def _read_workspace_marker(marker: Path) -> dict:
    if not marker.is_file():
        return {}
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_workspace_marker(
    context_dir: Path,
    workspace: Path,
    *,
    hidden_from_recent: bool = False,
    touch_opened: bool = False,
) -> None:
    marker = context_dir / "workspace.json"
    data = _read_workspace_marker(marker)
    data["path"] = str(workspace.expanduser().resolve())
    data["hidden_from_recent"] = hidden_from_recent
    if touch_opened:
        data["last_opened_at"] = datetime.now(timezone.utc).isoformat()
    try:
        context_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def _recent_sort_key(marker_data: dict, session_updated: str | None) -> str:
    opened = str(marker_data.get("last_opened_at") or "")
    session = session_updated or ""
    if opened and session:
        return max(opened, session)
    return opened or session


def _workspace_record(ctx_dir: Path, slug: str, ws_path: str) -> WorkspaceRecord:
    sessions_dir = ctx_dir / "sessions"
    session_count = 0
    plan_count = 0
    latest: str | None = None
    if sessions_dir.is_dir():
        for sess in sessions_dir.iterdir():
            if not sess.is_dir():
                continue
            session_count += 1
            if sess.name.startswith("plan-"):
                plan_count += 1
            mtime = _mtime_iso(sess / "plan_state.json") or _mtime_iso(sess / "messages.jsonl")
            if mtime and (latest is None or mtime > latest):
                latest = mtime
    return WorkspaceRecord(
        slug=slug,
        path=ws_path,
        session_count=session_count,
        plan_count=plan_count,
        updated_at=latest,
    )


def _hide_duplicate_workspace_markers(workspace: Path, *, keep_slug: str) -> None:
    """同一路径只保留 canonical slug，其余 context 目录从最近列表隐藏。"""
    ctx_root = resolve_context_root()
    if not ctx_root.is_dir():
        return
    try:
        resolved = str(workspace.expanduser().resolve())
    except OSError:
        return
    for child in ctx_root.iterdir():
        if not child.is_dir() or child.name == keep_slug:
            continue
        data = _read_workspace_marker(child / "workspace.json")
        path = str(data.get("path") or "").strip()
        if not path:
            continue
        try:
            if str(Path(path).expanduser().resolve()) == resolved:
                _write_workspace_marker(
                    child,
                    Path(path),
                    hidden_from_recent=True,
                )
        except OSError:
            continue


def list_workspaces() -> list[WorkspaceRecord]:
    """
    扫描 context 下所有工作区（跳过已从最近列表移除的项）。

    同一路径只保留 canonical slug；临时目录与无效 path 不展示。

    @return WorkspaceRecord 列表，按最近活跃排序
    """
    ctx_root = resolve_context_root()
    if not ctx_root.is_dir():
        return []

    merged: dict[str, tuple[str, WorkspaceRecord]] = {}

    for child in sorted(ctx_root.iterdir()):
        if not child.is_dir():
            continue
        slug = child.name
        marker = child / "workspace.json"
        data = _read_workspace_marker(marker)
        if data.get("hidden_from_recent"):
            continue
        ws_path = str(data.get("path") or "").strip()
        if not ws_path:
            continue
        try:
            ws = Path(ws_path).expanduser()
            try:
                resolved = str(ws.resolve())
            except OSError:
                resolved = str(ws)
            if is_ephemeral_workspace_path(resolved):
                _write_workspace_marker(child, ws, hidden_from_recent=True)
                continue
            if is_packaged_example_workspace(resolved):
                _write_workspace_marker(child, ws, hidden_from_recent=True)
                continue
            if not ws.is_dir():
                continue
        except OSError:
            continue

        canonical_slug = workspace_context_slug(ws)
        canonical_ctx = ctx_root / canonical_slug
        if slug != canonical_slug:
            _write_workspace_marker(
                canonical_ctx,
                ws,
                hidden_from_recent=False,
                touch_opened=False,
            )
            _write_workspace_marker(
                child,
                ws,
                hidden_from_recent=True,
            )

        marker_data = _read_workspace_marker(canonical_ctx / "workspace.json")
        record = _workspace_record(canonical_ctx, canonical_slug, resolved)
        sort_key = _recent_sort_key(marker_data, record.updated_at)
        prev = merged.get(resolved)
        if prev is None or sort_key > prev[0]:
            merged[resolved] = (sort_key, record)

    items = sorted(merged.values(), key=lambda item: item[0], reverse=True)
    return [record for _, record in items]


def resolve_workspace_path(slug: str) -> str:
    """
    由 slug 解析工作区根路径。

    @param slug context 目录名
    @return 工作区绝对路径
    """
    ctx_dir = resolve_context_root() / slug
    if not ctx_dir.is_dir():
        raise FileNotFoundError(f"工作区不存在: {slug}")
    marker = ctx_dir / "workspace.json"
    data = _read_workspace_marker(marker)
    if not data:
        raise FileNotFoundError(f"缺少 workspace.json: {slug}")
    path = str(data.get("path") or "").strip()
    if not path:
        raise FileNotFoundError(f"workspace.json 无 path: {slug}")
    ws = Path(path).expanduser()
    if not ws.is_dir():
        raise FileNotFoundError(f"工作区路径不存在: {path}")
    return str(ws.resolve())


def register_workspace(path: str) -> WorkspaceRecord:
    """
    注册工作区到 context（写入 workspace.json）。

    @param path 工作区绝对路径
    @return WorkspaceRecord
    """
    ws = Path(path).expanduser().resolve()
    if not ws.is_dir():
        raise FileNotFoundError(f"不是有效目录: {path}")
    if is_ephemeral_workspace_path(str(ws)):
        raise ValueError("不能用系统临时目录作为工作区，请选择真实项目目录")
    if is_packaged_example_workspace(str(ws)):
        raise ValueError("不能使用 llgraph 包内 examples 模板目录作为工作区")
    slug = workspace_context_slug(ws)
    ctx_dir = resolve_context_root() / slug
    _write_workspace_marker(
        ctx_dir,
        ws,
        hidden_from_recent=False,
        touch_opened=True,
    )
    _hide_duplicate_workspace_markers(ws, keep_slug=slug)
    return _workspace_record(ctx_dir, slug, str(ws))


def touch_workspace_opened(slug: str) -> None:
    """
    记录工作区最近打开时间（用于最近列表排序）。

    @param slug context 目录名
    """
    ctx_dir = resolve_context_root() / slug
    marker = ctx_dir / "workspace.json"
    data = _read_workspace_marker(marker)
    if not data:
        raise FileNotFoundError(f"工作区不存在: {slug}")
    path = str(data.get("path") or "").strip()
    if not path:
        raise FileNotFoundError(f"workspace.json 无 path: {slug}")
    data["last_opened_at"] = datetime.now(timezone.utc).isoformat()
    try:
        marker.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def dismiss_workspace_from_recent(slug: str) -> None:
    """
    从最近工作区列表隐藏（不删除 sessions 等数据）。

    @param slug context 目录名
    """
    ctx_dir = resolve_context_root() / slug
    marker = ctx_dir / "workspace.json"
    data = _read_workspace_marker(marker)
    if not data:
        raise FileNotFoundError(f"工作区不存在: {slug}")
    path = str(data.get("path") or "").strip()
    if not path:
        raise FileNotFoundError(f"workspace.json 无 path: {slug}")
    _write_workspace_marker(
        ctx_dir,
        Path(path),
        hidden_from_recent=True,
    )
