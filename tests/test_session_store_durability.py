"""会话落盘耐久性：写被打断不毁会话，单行坏掉不丢整段历史。

原来 `messages.jsonl` 是就地截断再逐行写，读侧一遇坏行整份丢弃：
写到一半被 Ctrl-C / SIGTERM / 磁盘写满打断，下次续聊就是空会话。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from llgraph.session import session_file_store
from llgraph.session.atomic_store import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    is_temp_sibling_path,
    sweep_stale_temp_files,
)
from llgraph.session.session_file_store import (
    CORRUPT_BACKUP_PREFIX,
    load_session_messages,
    quarantine_corrupt_messages,
    read_session_message_rows,
    save_session_messages,
)
from llgraph.session.user_storage import session_messages_path

THREAD = "cli-durability"


def _turns(count: int, *, body: str = "") -> list:
    out: list = []
    for i in range(count):
        out.append(HumanMessage(content=f"问题 {i}{body}"))
        out.append(AIMessage(content=f"回答 {i}{body}"))
    return out


class _Unserializable:
    """json.dumps(default=str) 也救不了的对象（__str__ 抛错）。"""

    def __str__(self) -> str:
        raise TypeError("nope")


class _RaisingRows:
    """迭代到第二行才失败，模拟写一半断掉。"""

    def __iter__(self):
        yield {"ok": 1}
        raise OSError("disk full")


def test_atomic_write_leaves_original_on_failure(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    atomic_write_text(path, "第一版\n")

    with pytest.raises(OSError):
        atomic_write_jsonl(path, _RaisingRows())
    assert path.read_text(encoding="utf-8") == "第一版\n"

    with pytest.raises(TypeError):
        atomic_write_json(path, _Unserializable())
    assert path.read_text(encoding="utf-8") == "第一版\n"

    assert not [p for p in tmp_path.iterdir() if is_temp_sibling_path(p)], "失败后不该留 tmp"


def test_save_session_messages_is_atomic(tmp_path: Path) -> None:
    save_session_messages(tmp_path, THREAD, _turns(3))
    path = session_messages_path(tmp_path, THREAD)
    assert path.is_file()

    # 写新一版时中途失败：磁盘上必须还是完整旧版，而不是半截新版
    def explode(*_a: object, **_k: object) -> None:
        raise OSError("disk full")

    original = session_file_store.atomic_write_jsonl
    session_file_store.atomic_write_jsonl = explode  # type: ignore[assignment]
    try:
        assert save_session_messages(tmp_path, THREAD, _turns(9)) is None
    finally:
        session_file_store.atomic_write_jsonl = original  # type: ignore[assignment]

    rows, dropped = read_session_message_rows(path)
    assert dropped == 0
    assert len(rows) == 6


def test_truncated_tail_line_keeps_earlier_history(tmp_path: Path) -> None:
    save_session_messages(tmp_path, THREAD, _turns(4))
    path = session_messages_path(tmp_path, THREAD)
    raw = path.read_text(encoding="utf-8")

    # 模拟旧版非原子写被打断：最后一行只落了一半
    truncated = raw[: -len(raw.splitlines()[-1]) // 2]
    path.write_text(truncated, encoding="utf-8")

    loaded = load_session_messages(tmp_path, THREAD)
    assert len(loaded) >= 7, "只该丢半截的那一条，不该整段清空"
    assert loaded[0].content == "问题 0"

    # 读到坏行会顺手改写回干净的一份，下次读不再有坏行
    _rows, dropped = read_session_message_rows(path)
    assert dropped == 0


def test_unrecoverable_history_is_quarantined_before_overwrite(tmp_path: Path) -> None:
    """整份读不出来也不能静默清空：原始字节先留一份，本轮覆盖写才不算毁数据。"""
    path = session_messages_path(tmp_path, THREAD)
    path.parent.mkdir(parents=True, exist_ok=True)
    garbage = '{"type":"human","data":{\n{"nope"\n'
    path.write_text(garbage, encoding="utf-8")

    assert load_session_messages(tmp_path, THREAD) == []

    backups = [p for p in path.parent.iterdir() if CORRUPT_BACKUP_PREFIX in p.name]
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == garbage


def test_quarantine_keeps_at_most_three_backups(tmp_path: Path) -> None:
    path = session_messages_path(tmp_path, THREAD)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("坏字节", encoding="utf-8")
    for i in range(5):
        stale = path.with_name(f"{path.name}.{CORRUPT_BACKUP_PREFIX}2026010{i}T000000")
        stale.write_text(f"old {i}", encoding="utf-8")

    fresh = quarantine_corrupt_messages(path)
    assert fresh is not None and fresh.read_text(encoding="utf-8") == "坏字节"

    kept = sorted(p for p in path.parent.iterdir() if CORRUPT_BACKUP_PREFIX in p.name)
    assert len(kept) == 3, "副本不能无限堆积"
    assert fresh in kept
    assert [p.read_text(encoding="utf-8") for p in kept[:2]] == ["old 3", "old 4"], "只留最近的"


def test_garbage_middle_line_drops_only_that_line(tmp_path: Path) -> None:
    save_session_messages(tmp_path, THREAD, _turns(3))
    path = session_messages_path(tmp_path, THREAD)
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[2] = '{"type":"human","data":{'
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows, dropped = read_session_message_rows(path)
    assert dropped == 1
    assert len(rows) == 5


def test_concurrent_writer_and_reader_never_see_partial_file(tmp_path: Path) -> None:
    """CLI 落盘的同时 Web / 会话列表在读同一份 messages.jsonl。

    正文要大到超过文本缓冲（真实会话动辄几百 KB），旧的就地截断写才会真的
    在 close 之前多次 write，读侧看到的就是半截文件。
    """
    path = session_messages_path(tmp_path, THREAD)
    counts = (2, 5, 8, 11)
    valid_lengths = {2 * n for n in counts}
    payloads = {n: _turns(n, body="内容" * 4000) for n in counts}
    save_session_messages(tmp_path, THREAD, payloads[2])

    errors: list[str] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            rows, dropped = read_session_message_rows(path)
            if dropped:
                errors.append(f"读到 {dropped} 条坏行")
            elif rows and len(rows) not in valid_lengths:
                errors.append(f"读到不完整版本：{len(rows)} 行")

    def writer(turns: int) -> None:
        for _ in range(15):
            save_session_messages(tmp_path, THREAD, payloads[turns])

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()
    writers = [threading.Thread(target=writer, args=(n,)) for n in counts]
    for t in writers:
        t.start()
    for t in writers:
        t.join()
    stop.set()
    reader_thread.join(timeout=5)

    assert not errors, errors[:3]
    rows, dropped = read_session_message_rows(path)
    assert dropped == 0
    assert len(rows) in valid_lengths, "落盘的必须是某个写者的完整版本"
    assert not [p for p in path.parent.iterdir() if is_temp_sibling_path(p)]


def test_load_after_process_kill_between_writes(tmp_path: Path) -> None:
    """替换前被 kill：目标文件仍是上一版，残留 tmp 不影响读，且能被清理。"""
    save_session_messages(tmp_path, THREAD, _turns(2))
    path = session_messages_path(tmp_path, THREAD)
    orphan = path.with_name(f".{path.name}.dead-tmp.tmp")
    orphan.write_text('{"type":"human","data":{', encoding="utf-8")

    loaded = load_session_messages(tmp_path, THREAD)
    assert len(loaded) == 4

    assert sweep_stale_temp_files(path.parent, min_age_sec=0.0) == 1
    assert not orphan.exists()
    assert path.is_file()


def test_sweep_spares_tmp_files_other_processes_may_be_writing(tmp_path: Path) -> None:
    fresh = tmp_path / ".messages.jsonl.other-proc.tmp"
    fresh.write_text("half", encoding="utf-8")
    plain = tmp_path / "keep.json"
    plain.write_text("{}", encoding="utf-8")

    assert sweep_stale_temp_files(tmp_path, min_age_sec=3600.0) == 0
    assert fresh.exists() and plain.exists()


def test_todo_state_survives_concurrent_writes(tmp_path: Path) -> None:
    from llgraph.core.todo_store import TodoItem, TodoState, load_todo_state, save_todo_state
    from llgraph.session.user_storage import session_todos_path

    def writer(n: int) -> None:
        state = TodoState(
            todos=[TodoItem(id=f"t{i}", content=f"任务 {i}", status="pending") for i in range(n)],
            updated_at="",
        )
        for _ in range(12):
            save_todo_state(tmp_path, THREAD, state)

    threads = [threading.Thread(target=writer, args=(n,)) for n in (1, 3, 5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    path = session_todos_path(tmp_path, THREAD)
    json.loads(path.read_text(encoding="utf-8"))
    assert len(load_todo_state(tmp_path, THREAD).todos) in {1, 3, 5}
