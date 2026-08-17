"""会话历史检索滥用拦截。"""

from __future__ import annotations

from pathlib import Path

from llgraph.session.session_history_search import guard_session_history_query


def test_guard_blocks_meta_query(tmp_path: Path) -> None:
    msg = (
        '{"type":"human","data":{"content":"hello","type":"human"}}\n'
    )
    # session layout via monkeypatch paths - write under tmp and patch helpers
    from llgraph.session import session_history_search as mod

    thread = "cli-test"
    messages = tmp_path / "messages.jsonl"
    messages.write_text(msg, encoding="utf-8")

    def fake_messages(_ws: Path, _tid: str) -> Path:
        return messages

    def fake_archive(_ws: Path, _tid: str) -> Path:
        return tmp_path / "missing_archive.jsonl"

    orig_m = mod.session_messages_path
    orig_a = mod.session_archive_jsonl_path
    mod.session_messages_path = fake_messages  # type: ignore[assignment]
    mod.session_archive_jsonl_path = fake_archive  # type: ignore[assignment]
    try:
        blocked = guard_session_history_query(tmp_path, thread, "用户异议 上一轮结论")
        assert blocked is not None
        assert "拒绝" in blocked or "元叙事" in blocked
    finally:
        mod.session_messages_path = orig_m  # type: ignore[assignment]
        mod.session_archive_jsonl_path = orig_a  # type: ignore[assignment]


def test_guard_blocks_single_turn_no_archive(tmp_path: Path) -> None:
    from llgraph.session import session_history_search as mod

    messages = tmp_path / "messages.jsonl"
    messages.write_text(
        '{"type":"human","data":{"content":"埋点删除","type":"human"}}\n',
        encoding="utf-8",
    )

    def fake_messages(_ws: Path, _tid: str) -> Path:
        return messages

    def fake_archive(_ws: Path, _tid: str) -> Path:
        return tmp_path / "nope.jsonl"

    orig_m = mod.session_messages_path
    orig_a = mod.session_archive_jsonl_path
    mod.session_messages_path = fake_messages  # type: ignore[assignment]
    mod.session_archive_jsonl_path = fake_archive  # type: ignore[assignment]
    try:
        blocked = guard_session_history_query(tmp_path, "cli-x", "埋点 属性 删除")
        assert blocked is not None
        assert "只有本轮" in blocked or "无多轮" in blocked
    finally:
        mod.session_messages_path = orig_m  # type: ignore[assignment]
        mod.session_archive_jsonl_path = orig_a  # type: ignore[assignment]
