"""会话状态原子落盘：临时文件 + os.replace。

会话正文 / meta / manifest / 锚点 / 编辑账本原来都是 `open("w")` 就地截断再逐行写。
中途被 Ctrl-C、SIGTERM、OOM、磁盘写满打断，磁盘上留下的就是半截文件；
而读侧一遇到坏行就整份丢弃，等于一次意外退出报废整个会话。

这里统一成「写同目录临时文件 → os.replace 覆盖」：
os.replace 在 POSIX / Windows 上都是原子替换，读侧任何时刻看到的都是完整的旧版或完整的新版。
临时文件名带 pid + 线程 id + 计数，CLI 与 Web 同时写同一会话也不会互相踩。
"""

from __future__ import annotations

import itertools
import json
import os
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_TMP_SUFFIX = ".tmp"
_STALE_TMP_MIN_AGE_SEC = 3600.0
_tmp_counter = itertools.count()
_swept_dirs: set[str] = set()
_sweep_lock = threading.Lock()


def temp_sibling_path(path: Path) -> Path:
    """
    生成同目录唯一临时文件路径。

    同目录是硬要求：os.replace 跨文件系统会抛 OSError，
    /tmp 与 ~/.llgraph 很可能不在同一挂载点。

    @param path 目标文件路径
    @return 临时文件路径（隐藏文件，不会被 `*.jsonl` / `*.json` 之类的 glob 命中）
    """
    token = f"{os.getpid():x}-{threading.get_ident():x}-{next(_tmp_counter):x}"
    return path.with_name(f".{path.name}.{token}{_TMP_SUFFIX}")


def is_temp_sibling_path(path: Path) -> bool:
    """
    @param path 待判定路径
    @return 是否为本模块产生的临时文件（崩溃后可能残留）
    """
    name = path.name
    return name.startswith(".") and name.endswith(_TMP_SUFFIX)


def atomic_write_text(path: Path, text: str, *, fsync: bool = False) -> None:
    """
    原子写入文本文件。

    @param path 目标路径（父目录自动创建）
    @param text 全量正文
    @param fsync 是否 fsync（掉电也不丢，代价是一次磁盘同步）
    @raises OSError 写入或替换失败；此时目标文件保持原样
    """
    _atomic_write(path, lambda handle: handle.write(text), fsync=fsync)


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    indent: int | None = 2,
    fsync: bool = False,
) -> None:
    """
    原子写入 JSON 文件。

    序列化在写 tmp 的过程中进行，序列化失败同样不会碰到目标文件。

    @param path 目标路径
    @param payload 可 JSON 序列化对象
    @param indent 缩进；None 为紧凑
    @param fsync 是否 fsync
    @raises OSError 写入失败
    @raises TypeError 或 ValueError 序列化失败
    """
    text = json.dumps(payload, ensure_ascii=False, indent=indent, default=str)
    atomic_write_text(path, text, fsync=fsync)


def atomic_write_jsonl(
    path: Path,
    rows: Iterable[Any],
    *,
    fsync: bool = False,
) -> None:
    """
    原子写入 jsonl 文件（逐行流式写 tmp，不在内存里拼整份）。

    @param path 目标路径
    @param rows 每行一个可 JSON 序列化对象
    @param fsync 是否 fsync
    @raises OSError 写入失败
    @raises TypeError 或 ValueError 序列化失败
    """

    def _dump(handle: Any) -> None:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str))
            handle.write("\n")

    _atomic_write(path, _dump, fsync=fsync)


def _atomic_write(path: Path, dump: Any, *, fsync: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _sweep_once_per_process(path.parent)
    tmp = temp_sibling_path(path)
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            dump(handle)
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        # 半截 tmp 不留在会话目录里；替换没发生，目标文件仍是上一个完整版本
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _sweep_once_per_process(directory: Path) -> None:
    """本进程首次写某目录时顺手清一遍上次崩溃留下的 tmp（每目录仅一次）。"""
    key = str(directory)
    with _sweep_lock:
        if key in _swept_dirs:
            return
        _swept_dirs.add(key)
    sweep_stale_temp_files(directory, min_age_sec=_STALE_TMP_MIN_AGE_SEC)


def sweep_stale_temp_files(directory: Path, *, min_age_sec: float = _STALE_TMP_MIN_AGE_SEC) -> int:
    """
    清掉目录里过期的残留临时文件（进程被 kill 在 replace 之前时产生）。

    只扫一层、只删够老的，避免误删同机其它进程正在写的 tmp。

    @param directory 待清理目录
    @param min_age_sec 最小存活秒数
    @return 删除个数
    """
    if not directory.is_dir():
        return 0
    now = time.time()
    removed = 0
    try:
        entries = list(directory.iterdir())
    except OSError:
        return 0
    for item in entries:
        if not is_temp_sibling_path(item):
            continue
        try:
            if not item.is_file() or now - item.stat().st_mtime < min_age_sec:
                continue
            item.unlink()
            removed += 1
        except OSError:
            continue
    return removed
