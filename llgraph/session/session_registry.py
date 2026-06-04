"""会话发现、列举与恢复（~/.llgraph/context/.../messages.jsonl）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from llgraph.session.session_file_store import session_has_messages_file
from llgraph.session.session_meta import (
    backfill_session_titles,
    get_session_title,
    resolve_session_display_title,
)
from llgraph.session.user_storage import (
    format_storage_location_hint,
    legacy_workspace_session_dir,
    session_edits_path,
    session_messages_path,
    session_thread_dir,
    user_sessions_root,
)


@dataclass(frozen=True)
class SessionSummary:
    """可恢复会话摘要。"""

    thread_id: str
    title: str
    title_is_stored: bool
    updated_at: str | None
    message_count: int
    has_manifest: bool
    has_archive: bool
    has_edits: bool
    workspace_hint: str | None
    sources: tuple[str, ...]


def _parse_iso(value: str | None) -> str | None:
    if not value or not str(value).strip():
        return None
    return str(value).strip()


def _mtime_iso(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except OSError:
        return None


def _read_edit_meta(workspace: Path, thread_id: str) -> tuple[str | None, str | None]:
    meta_path = session_thread_dir(workspace, thread_id) / "meta.json"
    if not meta_path.is_file():
        legacy_meta = legacy_workspace_session_dir(workspace, thread_id) / "meta.json"
        if legacy_meta.is_file():
            meta_path = legacy_meta
        else:
            return None, None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _mtime_iso(meta_path), None
    started = _parse_iso(data.get("started_at"))
    ws = data.get("workspace")
    ws_str = str(ws).strip() if ws else None
    return started, ws_str


def _discover_disk_session_ids(workspace: Path) -> set[str]:
    ids: set[str] = set()
    root = workspace.expanduser().resolve()

    legacy_root = root / ".llgraph" / "sessions"
    if legacy_root.is_dir():
        for child in legacy_root.iterdir():
            if child.is_dir():
                ids.add(child.name)

    user_ctx = user_sessions_root(root)
    if user_ctx.is_dir():
        for child in user_ctx.iterdir():
            if child.is_dir():
                ids.add(child.name)
        for path in user_ctx.glob("*.jsonl"):
            ids.add(path.stem)

    return ids


def _count_messages_file(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _load_jsonl_sessions(workspace: Path) -> dict[str, tuple[int, str | None]]:
    """
    从各会话 messages.jsonl 统计条数。

    @return thread_id -> (count, updated_at)
    """
    result: dict[str, tuple[int, str | None]] = {}
    root = user_sessions_root(workspace.expanduser().resolve())
    if not root.is_dir():
        return result
    for child in root.iterdir():
        if child.is_dir():
            msg_path = child / "messages.jsonl"
            if msg_path.is_file():
                result[child.name] = (_count_messages_file(msg_path), _mtime_iso(msg_path))
    return result


def _session_artifact_paths(workspace: Path, thread_id: str) -> tuple[Path, Path, Path]:
    root = workspace.expanduser().resolve()
    user_ctx = user_sessions_root(root)
    manifest_path = user_ctx / thread_id / "manifest.json"
    archive_path = user_ctx / f"{thread_id}.jsonl"
    edits_dir = session_thread_dir(root, thread_id)
    return manifest_path, archive_path, edits_dir


def list_workspace_session_ids(
    workspace: Path,
    *,
    include_empty: bool = True,
) -> list[str]:
    """
    列举工作区下所有会话 thread_id（不触发标题回填）。

    @param workspace 工作区根
    @param include_empty 是否包含仅 manifest/meta 的空壳会话
    @return 排序后的 ID 列表
    """
    root = workspace.expanduser().resolve()
    disk_ids = _discover_disk_session_ids(root)
    msg_map = _load_jsonl_sessions(root)
    all_ids = sorted(disk_ids | set(msg_map.keys()))
    if include_empty:
        return all_ids
    return [tid for tid in all_ids if session_has_substantive_content(root, tid)]


def session_has_substantive_content(workspace: Path, thread_id: str) -> bool:
    """
    是否为「有实质内容」的会话（非仅启动/列表产生的空壳目录）。

    @param workspace 工作区根
    @param thread_id 线程 ID
    @return 是否有对话、归档或编辑记录
    """
    root = workspace.expanduser().resolve()
    msg_path = session_messages_path(root, thread_id)
    if msg_path.is_file() and _count_messages_file(msg_path) > 0:
        return True
    if session_edits_path(root, thread_id).is_file():
        return True
    legacy_edits = legacy_workspace_session_dir(root, thread_id) / "edits.jsonl"
    if legacy_edits.is_file():
        return True
    _, archive_path, _ = _session_artifact_paths(root, thread_id)
    if archive_path.is_file() and _count_messages_file(archive_path) > 0:
        return True
    return False


def list_empty_session_ids(workspace: Path) -> list[str]:
    """
    列举无实质内容的空壳会话 ID。

    @param workspace 工作区根
    @return 空壳会话 ID 列表
    """
    root = workspace.expanduser().resolve()
    disk_ids = _discover_disk_session_ids(root)
    msg_map = _load_jsonl_sessions(root)
    all_ids = sorted(disk_ids | set(msg_map.keys()))
    return [tid for tid in all_ids if not session_has_substantive_content(root, tid)]


def discover_sessions(workspace: Path) -> list[SessionSummary]:
    """
    合并 messages.jsonl 与磁盘目录，列举工作区相关会话。

    @param workspace 工作区根
    @return 按最近活动排序的会话列表
    """
    root = workspace.expanduser().resolve()
    disk_ids = _discover_disk_session_ids(root)
    msg_map = _load_jsonl_sessions(root)
    all_ids = disk_ids | set(msg_map.keys())
    listable_ids = [
        tid for tid in sorted(all_ids) if session_has_substantive_content(root, tid)
    ]
    backfill_session_titles(root, listable_ids)

    summaries: list[SessionSummary] = []
    for tid in listable_ids:
        sources: list[str] = []
        msg_count, msg_updated = msg_map.get(tid, (0, None))
        if msg_count > 0:
            sources.append("jsonl")
        if tid in disk_ids:
            sources.append("disk")

        manifest_path, archive_path, edits_dir = _session_artifact_paths(root, tid)
        has_manifest = manifest_path.is_file()
        has_archive = archive_path.is_file()
        legacy_edits = legacy_workspace_session_dir(root, tid) / "edits.jsonl"
        has_edits = (
            session_edits_path(root, tid).is_file()
            or legacy_edits.is_file()
            or (edits_dir / "meta.json").is_file()
        )

        started, ws_hint = _read_edit_meta(root, tid)
        if msg_count == 0 and session_has_messages_file(root, tid):
            msg_count = _count_messages_file(session_messages_path(root, tid))
            msg_updated = _mtime_iso(session_messages_path(root, tid))

        updated_candidates = [
            msg_updated,
            _mtime_iso(manifest_path),
            _mtime_iso(archive_path),
            _mtime_iso(edits_dir / "edits.jsonl"),
            started,
        ]
        updated_at = max((x for x in updated_candidates if x), default=None)

        stored_title = get_session_title(root, tid)
        display_title = (
            stored_title
            if stored_title
            else resolve_session_display_title(root, tid)
        )
        summaries.append(
            SessionSummary(
                thread_id=tid,
                title=display_title,
                title_is_stored=bool(stored_title),
                updated_at=updated_at,
                message_count=msg_count,
                has_manifest=has_manifest,
                has_archive=has_archive,
                has_edits=has_edits,
                workspace_hint=ws_hint,
                sources=tuple(sources),
            )
        )

    summaries.sort(
        key=lambda s: s.updated_at or "",
        reverse=True,
    )
    return summaries


def format_sessions_list(workspace: Path, *, current_thread_id: str | None = None) -> str:
    """
    格式化 /sessions 输出。

    @param workspace 工作区根
    @param current_thread_id 当前会话 ID（标记 ●）
    @return 多行文本
    """
    from llgraph.core.checkpointer_factory import checkpointer_kind

    sessions = discover_sessions(workspace)
    empty_ids = list_empty_session_ids(workspace)
    kind = checkpointer_kind(workspace, with_memory=True)
    lines = [
        "会话列表（标题 · thread_id）",
        f"记忆持久化: {kind}（messages.jsonl 可跨重启恢复；可直接打开查看/编辑）",
        format_storage_location_hint(workspace),
        "",
    ]
    if empty_ids:
        lines.append(
            f"（另有 {len(empty_ids)} 个空壳会话未展示，可用 /session delete empty 清理）"
        )
        lines.append("")
    if not sessions:
        lines.append("  （无）启动交互后会生成 cli-xxxxxxxx。")
        lines.append("")
        lines.append("恢复: llgraph -C <工作区> --thread-id <id>")
        lines.append("会话内: /session use <id>  |  /session new")
        lines.append("改标题: /session title <新标题>")
        lines.append("删除:   /session delete all --including-current")
        return "\n".join(lines)

    for item in sessions:
        mark = "● " if current_thread_id and item.thread_id == current_thread_id else "  "
        flags: list[str] = []
        if item.message_count > 0:
            flags.append(f"msg={item.message_count}")
        if item.has_archive:
            flags.append("归档")
        if item.has_manifest:
            flags.append("manifest")
        if item.has_edits:
            flags.append("编辑")
        flag_text = ", ".join(flags) if flags else "仅目录"
        updated = item.updated_at[:19].replace("T", " ") if item.updated_at else "未知"
        title = item.title
        if len(title) > 36:
            title = title[:35] + "…"
        lines.append(
            f"{mark}{title:<18}  {item.thread_id}  更新 {updated}  ({flag_text})"
        )
    lines.append("")
    lines.append("恢复: llgraph -C <工作区> --thread-id <id>  或  /session use <id>")
    lines.append("新建: /session new")
    lines.append("改标题: /session title <新标题>  （当前会话）")
    lines.append("       /session title <id> <新标题>  （指定会话）")
    lines.append("删除:   /session delete <id>  |  /session delete all")
    lines.append("       /session delete empty  （仅删空壳）")
    lines.append("       /session delete all --including-current  （含当前，并切到新会话）")
    return "\n".join(lines)


def session_is_resumable(workspace: Path, thread_id: str) -> tuple[bool, str]:
    """
    判断 thread_id 是否可恢复。

    @return (可恢复, 说明)
    """
    root = workspace.expanduser().resolve()
    if session_has_messages_file(root, thread_id):
        return True, "将从 messages.jsonl 恢复对话状态。"
    disk_ids = _discover_disk_session_ids(root)
    if thread_id in disk_ids:
        return (
            True,
            "无 SQLite 对话检查点；将加载 manifest/归档锚点，对话正文需 read_file 归档或重新说明目标。",
        )
    return False, "未找到该会话记录，将按新会话启动（仅使用此 thread_id）。"
