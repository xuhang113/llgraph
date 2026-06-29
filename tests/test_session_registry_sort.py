"""session_registry 按最近活动排序。"""

from __future__ import annotations

from pathlib import Path

from llgraph.session.session_meta import save_session_meta, touch_session_activity
from llgraph.session.session_registry import discover_sessions, _max_iso
from llgraph.session.user_storage import session_messages_path


def test_max_iso_picks_latest_timestamp() -> None:
    assert _max_iso("2026-01-01T00:00:00Z", "2026-06-01T12:00:00+00:00") == "2026-06-01T12:00:00Z"


def test_discover_sessions_orders_by_meta_updated_at(tmp_path: Path) -> None:
    older = "cli-older01"
    newer = "cli-newer01"
    for tid in (older, newer):
        msg_path = session_messages_path(tmp_path, tid)
        msg_path.parent.mkdir(parents=True, exist_ok=True)
        msg_path.write_text('{"role":"user","content":"hi"}\n', encoding="utf-8")
    save_session_meta(
        tmp_path,
        older,
        {"session_kind": "agent", "title": "old", "updated_at": "2026-01-01T00:00:00Z"},
    )
    touch_session_activity(tmp_path, newer)

    ordered = [s.thread_id for s in discover_sessions(tmp_path)]
    assert ordered.index(newer) < ordered.index(older)
