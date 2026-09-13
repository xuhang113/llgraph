"""工作区源文件的原子落盘：同目录临时文件 + os.replace。

会话状态早已走 `llgraph.session.atomic_store`，但用户源码一直是
`Path.write_text()` —— 先把目标文件截断成 0 字节，再往里写。
在这中间被 Ctrl-C（用户掐掉跑飞的 Agent）、SIGTERM、OOM、磁盘写满打断，
磁盘上留下的就是空文件或半截文件，而这份代码可能没在 git 里。

与会话落盘不同，这里替换的是用户自己的文件，必须额外保住文件语义：

- 目标是符号链接时先解到真实文件，别把链接本身换成普通文件
- 保留权限位（`collect.sh` 0755 被换成 0644 就跑不起来了）
- 硬链接（st_nlink > 1）退回就地覆盖：os.replace 会切断链接关系
- 建不出临时文件（目录只读等）也退回就地覆盖，不能因此让写工具失去能力

不做 fsync：这里防的是进程被打断，os.replace 之后内容已在页缓存里，
掉电级别的持久化不值得每次编辑都付一次磁盘同步。
"""

from __future__ import annotations

import errno
import itertools
import os
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_TMP_SUFFIX = ".llgraph-tmp"
_STALE_TMP_MIN_AGE_SEC = 3600.0
_NEW_FILE_MODE = 0o666  # 与普通 open("w") 一致，实际权限由 umask 决定
_tmp_counter = itertools.count()
_swept_dirs: set[str] = set()
_sweep_lock = threading.Lock()

# 这些错误说明「在目标目录里建不出临时文件」，退回就地覆盖仍可能成功。
# 磁盘写满（ENOSPC）不在其中：那种情况就地覆盖只会先截断再失败，宁可直接报错保住原文件。
_FALLBACK_ERRNOS = frozenset(
    {
        errno.EACCES,
        errno.EPERM,
        errno.EROFS,
        errno.EXDEV,
        errno.ENOTDIR,
        errno.ENAMETOOLONG,
    }
)


@dataclass(frozen=True)
class WorkspaceWriteResult:
    """一次工作区写入的落盘方式（供测试与排障断言，不进模型上下文）。"""

    path: Path
    atomic: bool
    fallback_reason: str = ""


def temp_sibling_path(path: Path) -> Path:
    """
    生成同目录唯一临时文件路径。

    同目录是硬要求：os.replace 跨文件系统会抛 OSError。
    名字带 `.llgraph-tmp` 后缀，人不会这么命名文件，扫残留时不会误删用户文件。

    @param path 目标文件路径
    @return 临时文件路径（隐藏文件，`*.py` 之类 glob 不会命中）
    """
    token = f"{os.getpid():x}-{threading.get_ident():x}-{next(_tmp_counter):x}"
    return path.with_name(f".{path.name}.{token}{_TMP_SUFFIX}")


def is_temp_sibling_path(path: Path) -> bool:
    """
    @param path 待判定路径
    @return 是否为本模块产生的临时文件（进程被 kill 在 replace 之前时残留）
    """
    name = path.name
    return name.startswith(".") and name.endswith(_TMP_SUFFIX)


def write_workspace_text(path: Path, text: str) -> WorkspaceWriteResult:
    """
    写入工作区文本文件：能原子替换就原子替换，否则退回就地覆盖。

    失败时（除就地覆盖回退路径）目标文件保持原样，不会留下半截内容。

    @param path 目标文件路径（父目录自动创建）
    @param text 全量正文
    @return 落盘方式
    @raises OSError 写入失败；原子路径下目标文件仍是上一个完整版本
    """
    target = _resolve_symlink(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    st = _lstat_or_none(target)

    if st is not None and st.st_nlink > 1:
        _write_in_place(target, text)
        return WorkspaceWriteResult(target, atomic=False, fallback_reason="hardlink")

    _sweep_once_per_process(target.parent)
    tmp = temp_sibling_path(target)
    try:
        _write_new_file(tmp, text)
        if st is not None:
            _copy_file_metadata(st, tmp)
        os.replace(tmp, target)
    except OSError as exc:
        _unlink_quietly(tmp)
        if exc.errno in _FALLBACK_ERRNOS:
            _write_in_place(target, text)
            return WorkspaceWriteResult(
                target, atomic=False, fallback_reason=errno.errorcode.get(exc.errno, "oserror")
            )
        raise
    except BaseException:
        _unlink_quietly(tmp)
        raise
    return WorkspaceWriteResult(target, atomic=True)


def sweep_stale_temp_files(
    directory: Path, *, min_age_sec: float = _STALE_TMP_MIN_AGE_SEC
) -> int:
    """
    清掉目录里过期的残留临时文件，别把碎片留在用户仓库里被 git 看见。

    只扫一层、只删够老的，避免误删同机其它 llgraph 进程正在写的 tmp。

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


def _resolve_symlink(path: Path) -> Path:
    """符号链接解到真实文件，让写入落在链接指向的文件上而不是换掉链接。"""
    try:
        if not path.is_symlink():
            return path
        return Path(os.path.realpath(path))
    except OSError:
        return path


def _lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return path.stat()
    except OSError:
        return None


def _write_new_file(path: Path, text: str) -> None:
    """O_EXCL 建临时文件，权限交给 umask（与普通新建文件一致）。"""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _NEW_FILE_MODE)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _write_in_place(path: Path, text: str) -> None:
    """就地覆盖（不原子）：只在原子替换用不了时走这里。"""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _copy_file_metadata(src_stat: os.stat_result, tmp: Path) -> None:
    """把原文件的权限位（可执行位等）与属主搬到临时文件上；搬不动就算了。"""
    try:
        os.chmod(tmp, stat.S_IMODE(src_stat.st_mode))
    except OSError:
        pass
    geteuid = getattr(os, "geteuid", None)
    chown = getattr(os, "chown", None)
    if geteuid is None or chown is None:
        return
    if src_stat.st_uid == geteuid() and src_stat.st_gid == os.getegid():
        return
    try:
        chown(tmp, src_stat.st_uid, src_stat.st_gid)
    except OSError:
        pass


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _sweep_once_per_process(directory: Path) -> None:
    """本进程首次写某目录时顺手清一遍上次崩溃留下的 tmp（每目录仅一次）。"""
    key = str(directory)
    with _sweep_lock:
        if key in _swept_dirs:
            return
        _swept_dirs.add(key)
    sweep_stale_temp_files(directory)
