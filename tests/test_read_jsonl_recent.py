"""jsonl 末尾分页读取测试。"""

from __future__ import annotations

import json
from pathlib import Path

from llgraph.console.discovery import read_jsonl_lines, read_jsonl_lines_recent


def test_read_jsonl_lines_recent_returns_tail(tmp_path: Path) -> None:
    path = tmp_path / "messages.jsonl"
    for i in range(250):
        path.write_text(
            (path.read_text(encoding="utf-8") if path.exists() else "")
            + json.dumps({"i": i})
            + "\n",
            encoding="utf-8",
        )
    rows, total = read_jsonl_lines_recent(path, limit=200)
    assert total == 250
    assert len(rows) == 200
    assert rows[0]["i"] == 50
    assert rows[-1]["i"] == 249


def test_read_jsonl_lines_tolerates_invalid_utf8_bytes(tmp_path: Path) -> None:
    path = tmp_path / "messages.jsonl"
    path.write_bytes(
        b'{"type":"human","data":{"content":"ok"}}\n'
        b"\xff\xfe bad bytes\n"
        b'{"type":"ai","data":{"content":"reply"}}\n',
    )
    rows, total = read_jsonl_lines_recent(path, limit=10)
    assert total == 3
    assert len(rows) == 2
    assert rows[0]["type"] == "human"
    assert rows[-1]["type"] == "ai"


def test_read_jsonl_lines_head_vs_tail(tmp_path: Path) -> None:
    path = tmp_path / "messages.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for i in range(10):
            handle.write(json.dumps({"i": i}) + "\n")
    head, total = read_jsonl_lines(path, offset=0, limit=5)
    tail, total2 = read_jsonl_lines_recent(path, limit=5)
    assert total == total2 == 10
    assert head[0]["i"] == 0
    assert tail[-1]["i"] == 9
