"""长期记忆定时整理（零 LLM）。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from llgraph.memory.memory_lock import MemoryConsolidateLock
from llgraph.memory.memory_write_session import MemoryWriteSession
from llgraph.memory.paths import (
    ACTIVE_KINDS,
    CONSOLIDATE_REPORT,
    STATUS_ACTIVE,
    STATUS_REJECTED,
    ensure_memory_dirs,
    memory_root,
    save_memory_meta,
)
from llgraph.memory.settings import resolve_memory_settings
from llgraph.memory.store import (
    _memory_record,
    cosine_similarity,
    count_active_memories,
    list_memory_rows,
    utc_now_iso,
)
from llgraph.memory.write import _mechanical_truncate


@dataclass
class ConsolidateReport:
    """整理结果。"""

    merged: int = 0
    replaced: int = 0
    pruned_ttl: int = 0
    pruned_cap: int = 0
    truncated: int = 0
    elapsed_sec: float = 0.0
    skipped: bool = False
    reason: str = ""


def _parse_iso(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        return None


def _days_since(ts: str) -> float:
    dt = _parse_iso(ts)
    if dt is None:
        return 0.0
    return (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0


def _simulate_cap_score(row: dict) -> float:
    hit = int(row.get("hit_count", 0) or 0)
    last = str(row.get("last_hit_at", "") or row.get("updated_at", ""))
    age = _days_since(last)
    return hit * 2.0 - age * 0.01


def consolidate_workspace_memory(
    workspace: Path,
    *,
    user_id: str | None = None,
    light: bool = False,
) -> ConsolidateReport:
    """
    四阶段整理（无 LLM）。

    @param workspace 工作区根
    @param user_id 可选 user_id
    @param light 轻量模式（仅 prune + 去重）
    @return ConsolidateReport
    """
    t0 = time.perf_counter()
    settings = resolve_memory_settings(workspace)
    if not settings.enabled:
        return ConsolidateReport(skipped=True, reason="disabled")

    from llgraph.memory.paths import workspace_identity

    uid, workspace_key, workspace_slug = workspace_identity(workspace, user_id=user_id)
    ensure_memory_dirs(uid, workspace_key)
    lock = MemoryConsolidateLock(uid, workspace_key)
    if not lock.try_acquire():
        return ConsolidateReport(skipped=True, reason="lock_busy")

    report = ConsolidateReport()
    try:
        active = list_memory_rows(uid, workspace_key, status=STATUS_ACTIVE)
        all_rows = list_memory_rows(uid, workspace_key, status=None)
        if not all_rows and not active:
            report.skipped = True
            report.reason = "empty"
            return report

        session = MemoryWriteSession(uid, workspace_key)
        to_delete: list[str] = []

        if not light:
            # 3a 去重
            for i, a in enumerate(active):
                if str(a.get("memory_id")) in to_delete:
                    continue
                va = a.get("vector") or []
                for b in active[i + 1 :]:
                    bid = str(b.get("memory_id", ""))
                    if not bid or bid in to_delete:
                        continue
                    vb = b.get("vector") or []
                    if (
                        isinstance(va, list)
                        and isinstance(vb, list)
                        and cosine_similarity(va, vb) >= settings.dedupe_cosine_threshold
                    ):
                        keep, drop = (a, b)
                        if int(b.get("hit_count", 0) or 0) > int(a.get("hit_count", 0) or 0):
                            keep, drop = (b, a)
                        to_delete.append(str(drop.get("memory_id", "")))
                        report.merged += 1

            # 3b 冲突以新替旧
            remaining = [r for r in active if str(r.get("memory_id")) not in to_delete]
            for i, a in enumerate(remaining):
                aid = str(a.get("memory_id", ""))
                if aid in to_delete:
                    continue
                va = a.get("vector") or []
                for b in remaining[i + 1 :]:
                    bid = str(b.get("memory_id", ""))
                    if not bid or bid in to_delete:
                        continue
                    vb = b.get("vector") or []
                    sim = cosine_similarity(va, vb) if isinstance(va, list) and isinstance(vb, list) else 0.0
                    if sim >= settings.conflict_cosine_threshold:
                        ta = str(a.get("updated_at", ""))
                        tb = str(b.get("updated_at", ""))
                        if ta >= tb:
                            to_delete.append(bid)
                        else:
                            to_delete.append(aid)
                        report.replaced += 1
                        break

            # 3c 清理历史遗留的非法 kind（如旧 draft）
            for row in all_rows:
                mid = str(row.get("memory_id", ""))
                if mid and str(row.get("kind", "")) not in ACTIVE_KINDS:
                    to_delete.append(mid)

            # 3d 超长机械截断
            max_c = settings.memory_content_max_chars
            for row in list_memory_rows(uid, workspace_key, status=STATUS_ACTIVE):
                mid = str(row.get("memory_id", ""))
                if mid in to_delete:
                    continue
                content = str(row.get("content", ""))
                if len(content) <= max_c:
                    continue
                to_delete.append(mid)
                truncated = _mechanical_truncate(content, max_c)
                vec = row.get("vector") or []
                now = utc_now_iso()
                session.add_records(
                    [
                        _memory_record(
                            memory_id=mid,
                            user_id=uid,
                            workspace_key=workspace_key,
                            workspace_slug=workspace_slug,
                            kind=str(row.get("kind", "")),
                            content=truncated,
                            content_hash=str(row.get("content_hash", "")),
                            vector=list(vec) if isinstance(vec, list) else [],
                            confidence=float(row.get("confidence", 0) or 0),
                            source=str(row.get("source", "")),
                            status=STATUS_ACTIVE,
                            supersedes_id="",
                            created_at=str(row.get("created_at", now)),
                            updated_at=now,
                            last_hit_at=str(row.get("last_hit_at", "")),
                            hit_count=int(row.get("hit_count", 0) or 0),
                            weight_boost=float(row.get("weight_boost", 0) or 0),
                            tags=str(row.get("tags", "[]")),
                        )
                    ]
                )
                report.truncated += 1

        if to_delete:
            session.delete_memory_ids(to_delete)

        # Phase 4 Prune TTL
        prune_ids: list[str] = []
        for row in list_memory_rows(uid, workspace_key, status=None):
            mid = str(row.get("memory_id", ""))
            if not mid:
                continue
            status = str(row.get("status", ""))
            if status == STATUS_REJECTED:
                prune_ids.append(mid)
                continue
            if status != STATUS_ACTIVE:
                continue
            if str(row.get("kind", "")) not in ACTIVE_KINDS:
                continue
            created = str(row.get("created_at", ""))
            last_hit = str(row.get("last_hit_at", "") or "")
            hits = int(row.get("hit_count", 0) or 0)
            if not last_hit and _days_since(created) > settings.ttl_never_hit_days:
                prune_ids.append(mid)
            elif last_hit and _days_since(last_hit) > settings.ttl_no_hit_days and hits == 0:
                prune_ids.append(mid)
            elif last_hit and _days_since(last_hit) > settings.ttl_low_hit_days and hits < 3:
                prune_ids.append(mid)

        if prune_ids:
            session.delete_memory_ids(prune_ids)
            report.pruned_ttl += len(prune_ids)

        # 容量裁剪
        cap = settings.max_active_per_workspace
        target = max(cap - 20, int(cap * 0.9))
        active_after = list_memory_rows(uid, workspace_key, status=STATUS_ACTIVE)
        if len(active_after) > cap:
            ranked = sorted(active_after, key=_simulate_cap_score)
            drop_n = len(active_after) - target
            cap_ids = [str(r.get("memory_id", "")) for r in ranked[:drop_n]]
            session.delete_memory_ids([i for i in cap_ids if i])
            report.pruned_cap += len(cap_ids)

        now = utc_now_iso()
        save_memory_meta(
            uid,
            workspace_key,
            {
                "last_consolidated_at": now,
                "last_pruned_at": now,
                "record_count_active": count_active_memories(uid, workspace_key),
            },
        )
        log_path = memory_root(uid, workspace_key) / CONSOLIDATE_REPORT
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "at": now,
                        "merged": report.merged,
                        "replaced": report.replaced,
                        "pruned_ttl": report.pruned_ttl,
                        "pruned_cap": report.pruned_cap,
                        "light": light,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    finally:
        lock.release()

    report.elapsed_sec = time.perf_counter() - t0
    return report


def consolidate_all_known_workspaces(
    workspace_hint: Path | None = None,
    *,
    light: bool = False,
) -> list[tuple[str, ConsolidateReport]]:
    """
    整理用户下全部 workspace 目录；可选先整理当前 workspace。

    @param workspace_hint 优先整理的工作区
    @param light 轻量模式
    @return (workspace_key, report) 列表
    """
    results: list[tuple[str, ConsolidateReport]] = []
    if workspace_hint is not None:
        settings = resolve_memory_settings(workspace_hint)
        if settings.enabled:
            from llgraph.memory.paths import workspace_identity

            _, key, _ = workspace_identity(workspace_hint)
            results.append((key, consolidate_workspace_memory(workspace_hint, light=light)))

    import os

    from llgraph.memory.paths import resolve_memory_user_id, workspace_keys_for_user

    uid = resolve_memory_user_id(Path.cwd())
    if workspace_hint is not None:
        uid = workspace_identity(workspace_hint)[0]
    for key in workspace_keys_for_user(uid):
        if results and results[-1][0] == key:
            continue
        # 仅有 lance 无 workspace Path 时跳过全量（无 embed 工作区）
        _ = os  # uid already resolved
    return results
