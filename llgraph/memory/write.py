"""记忆写入、冲突替换、删除。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from llgraph.memory.embed import content_hash, embed_memory_text
from llgraph.memory.memory_write_session import MemoryWriteSession
from llgraph.memory.paths import ACTIVE_KINDS, KIND_PREF, STATUS_ACTIVE, workspace_identity
from llgraph.memory.settings import resolve_memory_settings
from llgraph.memory.store import (
    _memory_record,
    cosine_similarity,
    list_memory_rows,
    utc_now_iso,
)


@dataclass
class MemoryWriteReport:
    """写入结果。"""

    action: str
    memory_id: str
    kind: str
    content_preview: str
    replaced_ids: list[str]
    reason: str


def _clamp_content(text: str, max_chars: int) -> str:
    body = (text or "").strip()
    if len(body) <= max_chars:
        return body
    head = max_chars // 2
    tail = max_chars - head - 1
    return body[:head] + "…" + body[-tail:]


def _mechanical_truncate(text: str, max_chars: int) -> str:
    return _clamp_content(text, max_chars)


def upsert_memory(
    workspace: Path,
    *,
    content: str,
    kind: str = KIND_PREF,
    confidence: float = 0.9,
    source: str = "hot_tool",
    status: str = STATUS_ACTIVE,
    memory_id: str | None = None,
    replace_similar: bool = True,
) -> MemoryWriteReport:
    """
    写入或更新一条记忆。

    @param workspace 工作区根
    @param content 正文
    @param kind pref/fact/proc
    @param confidence 置信度
    @param source 来源
    @param status 状态
    @param memory_id 指定 ID 更新
    @param replace_similar 冲突时删旧
    @return MemoryWriteReport
    """
    settings = resolve_memory_settings(workspace)
    if not settings.enabled:
        return MemoryWriteReport("skip", "", kind, "", [], "memory_disabled")

    k = (kind or KIND_PREF).strip().lower()
    if k not in ACTIVE_KINDS:
        k = KIND_PREF

    user_id, workspace_key, workspace_slug = workspace_identity(workspace)
    body = _clamp_content(content, settings.memory_content_max_chars)
    if not body:
        return MemoryWriteReport("skip", "", k, "", [], "empty_content")

    chash = content_hash(body)
    vector = embed_memory_text(workspace, body)
    if not vector:
        return MemoryWriteReport("skip", "", k, "", [], "embed_failed")

    now = utc_now_iso()
    mid = (memory_id or "").strip() or str(uuid.uuid4())
    replaced: list[str] = []

    session = MemoryWriteSession(user_id, workspace_key)
    session._vector_dim = len(vector)
    if replace_similar:
        for row in list_memory_rows(user_id, workspace_key, status=STATUS_ACTIVE):
            if row.get("kind") != k:
                continue
            old_vec = row.get("vector") or []
            if isinstance(old_vec, list) and cosine_similarity(vector, old_vec) >= settings.conflict_cosine_threshold:
                old_id = str(row.get("memory_id", ""))
                if old_id and old_id != mid:
                    replaced.append(old_id)
        if replaced:
            session.delete_memory_ids(replaced)

    record = _memory_record(
        memory_id=mid,
        user_id=user_id,
        workspace_key=workspace_key,
        workspace_slug=workspace_slug,
        kind=k,
        content=body,
        content_hash=chash,
        vector=vector,
        confidence=confidence,
        source=source,
        status=status,
        supersedes_id=replaced[0] if replaced else "",
        created_at=now,
        updated_at=now,
        last_hit_at="",
        hit_count=0,
        weight_boost=0.0,
        tags="[]",
    )
    session.add_records([record])
    action = "replace" if replaced else "upsert"
    preview = body[:120] + ("…" if len(body) > 120 else "")
    reason = {
        "hot_tool": "user_explicit",
    }.get(source, source or "user_explicit")
    return MemoryWriteReport(action, mid, k, preview, replaced, reason)


def delete_memory(workspace: Path, memory_id: str) -> MemoryWriteReport:
    """
    按 ID 删除记忆。

    @param workspace 工作区根
    @param memory_id 记忆 ID
    @return MemoryWriteReport
    """
    settings = resolve_memory_settings(workspace)
    if not settings.enabled:
        return MemoryWriteReport("skip", memory_id, "", "", [], "memory_disabled")
    user_id, workspace_key, _ = workspace_identity(workspace)
    session = MemoryWriteSession(user_id, workspace_key)
    session.delete_memory_ids([memory_id.strip()])
    return MemoryWriteReport("delete", memory_id, "", "", [memory_id], "user_delete")


def format_memory_list(workspace: Path) -> str:
    """列出 active 记忆。"""
    settings = resolve_memory_settings(workspace)
    if not settings.enabled:
        return "长期记忆未启用（context.memory.enabled=false）。"
    user_id, workspace_key, _ = workspace_identity(workspace)
    lines = [f"长期记忆 user={user_id} workspace_key={workspace_key}", ""]
    rows = list_memory_rows(user_id, workspace_key, status=STATUS_ACTIVE)
    if not rows:
        lines.append("（暂无记忆）")
        return "\n".join(lines).strip()
    lines.append(f"## active ({len(rows)})")
    for row in rows[:50]:
        mid = str(row.get("memory_id", ""))[:8]
        kind = row.get("kind", "")
        content = str(row.get("content", ""))[:100]
        lines.append(f"- `{mid}…` [{kind}] {content}")
    return "\n".join(lines).strip()
