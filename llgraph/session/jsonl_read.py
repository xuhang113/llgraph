"""jsonl 容错读取：损坏 UTF-8 时用 replacement，避免 Web/CLI 500。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

_JSONL_ENCODING = "utf-8"
_JSONL_ERRORS = "replace"


def open_jsonl_for_read(path: Path):
    """
    以容错模式打开 jsonl 供读取。

    @param path jsonl 路径
    @return 文本文件句柄（调用方 with 管理）
    """
    return path.open(encoding=_JSONL_ENCODING, errors=_JSONL_ERRORS)


def iter_jsonl_text_lines(path: Path) -> Iterator[str]:
    """
    逐行 yield 非空 jsonl 行（已 strip）。

    @param path jsonl 路径
    @yield 单行 JSON 文本
    """
    if not path.is_file():
        return
    try:
        with open_jsonl_for_read(path) as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    yield stripped
    except OSError:
        return


def count_jsonl_lines(path: Path) -> int:
    """统计 jsonl 非空行数。"""
    return sum(1 for _ in iter_jsonl_text_lines(path))
