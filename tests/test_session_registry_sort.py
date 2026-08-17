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


def test_discover_sessions_ignores_web_trace_mtime(tmp_path: Path) -> None:
    """仅打开 Web 写入 trace 时，侧栏日期不应归入「今天」。"""
    import json
    import os
    import time
    from datetime import datetime, timezone

    from llgraph.session.session_meta import session_meta_json_path
    from llgraph.session.web_trace_store import live_web_trace_path

    tid = "cli-tracex1"
    msg_path = session_messages_path(tmp_path, tid)
    msg_path.parent.mkdir(parents=True, exist_ok=True)
    msg_path.write_text('{"role":"user","content":"hi"}\n', encoding="utf-8")
    old_ts = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc).timestamp()
    os.utime(msg_path, (old_ts, old_ts))

    meta_path = session_meta_json_path(tmp_path, tid)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {
                "session_kind": "agent",
                "title": "old session",
                "updated_at": "2026-01-15T10:00:00Z",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.utime(meta_path, (old_ts, old_ts))

    trace_path = live_web_trace_path(tmp_path, tid)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text('{"steps":[]}\n', encoding="utf-8")
    now_ts = time.time()
    os.utime(trace_path, (now_ts, now_ts))

    summary = next(s for s in discover_sessions(tmp_path) if s.thread_id == tid)
    assert summary.updated_at is not None
    assert summary.updated_at[:10] == "2026-01-15", summary.updated_at


def test_backfill_title_does_not_bump_updated_at(tmp_path: Path) -> None:
    """侧栏加载时补标题不应把旧会话刷进「今天」。"""
    import json
    import os
    from datetime import datetime, timezone

    from llgraph.session.session_meta import session_meta_json_path

    tid = "cli-title01"
    msg_path = session_messages_path(tmp_path, tid)
    msg_path.parent.mkdir(parents=True, exist_ok=True)
    msg_path.write_text(
        '{"role":"user","content":"Hi Hang, please help"}\n',
        encoding="utf-8",
    )
    old_ts = datetime(2026, 1, 10, 8, 0, 0, tzinfo=timezone.utc).timestamp()
    os.utime(msg_path, (old_ts, old_ts))

    meta_path = session_meta_json_path(tmp_path, tid)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {"session_kind": "agent", "updated_at": "2026-01-10T08:00:00Z"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.utime(meta_path, (old_ts, old_ts))

    summary = next(s for s in discover_sessions(tmp_path) if s.thread_id == tid)
    assert summary.title
    assert summary.updated_at is not None
    assert summary.updated_at[:10] == "2026-01-10", summary.updated_at
