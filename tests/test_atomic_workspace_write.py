"""改码写入原子化：被打断不留半截源文件，同时保住权限位 / 符号链接 / 硬链接。

老实现是 `Path.write_text()`：先截断目标文件再写。Ctrl-C 掐掉跑飞的 Agent、
SIGTERM、OOM 落在这中间，用户的源码就变成空文件或半截文件。
"""

from __future__ import annotations

import errno
import os
import stat
import time
from pathlib import Path

import pytest

from llgraph.core import atomic_write
from llgraph.core.atomic_write import (
    sweep_stale_temp_files,
    temp_sibling_path,
    write_workspace_text,
)
from llgraph.core.filesystem_tools import create_filesystem_tools
from llgraph.core.workspace import WorkspaceContext

_ORIGINAL = "def run():\n    return 1\n"


def _tool(root: Path, name: str):
    ctx = WorkspaceContext(root, allow_write=True)
    return next(t for t in create_filesystem_tools(ctx) if t.name == name)


def _leftover_temps(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name.endswith(".llgraph-tmp"))


def _file_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_interrupt_before_replace_keeps_original_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "app.py"
    target.write_text(_ORIGINAL, encoding="utf-8")

    def _boom(src, dst):
        raise KeyboardInterrupt

    monkeypatch.setattr(atomic_write.os, "replace", _boom)
    with pytest.raises(KeyboardInterrupt):
        write_workspace_text(target, "half written")

    assert target.read_text(encoding="utf-8") == _ORIGINAL
    assert _leftover_temps(tmp_path) == []


def test_search_replace_interrupt_does_not_truncate_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "app.py"
    target.write_text(_ORIGINAL, encoding="utf-8")
    tool = _tool(tmp_path, "search_replace")

    def _boom(src, dst):
        raise KeyboardInterrupt

    monkeypatch.setattr(atomic_write.os, "replace", _boom)
    with pytest.raises(KeyboardInterrupt):
        tool.invoke({"path": "app.py", "old_string": "return 1", "new_string": "return 2"})

    assert target.read_text(encoding="utf-8") == _ORIGINAL
    assert _leftover_temps(tmp_path) == []


def test_disk_full_reports_error_without_touching_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ENOSPC 不退回就地覆盖：就地覆盖只会先截断再失败，宁可报错保住原文件。"""
    target = tmp_path / "app.py"
    target.write_text(_ORIGINAL, encoding="utf-8")

    def _no_space(path: Path, text: str) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(atomic_write, "_write_new_file", _no_space)
    with pytest.raises(OSError):
        write_workspace_text(target, "new body")

    assert target.read_text(encoding="utf-8") == _ORIGINAL


def test_falls_back_to_in_place_when_temp_cannot_be_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """临时文件建不出来（目录只读等）也得写成功，不能让写工具直接失去能力。"""
    target = tmp_path / "app.py"
    target.write_text(_ORIGINAL, encoding="utf-8")

    def _denied(path: Path, text: str) -> None:
        raise PermissionError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(atomic_write, "_write_new_file", _denied)
    result = write_workspace_text(target, "new body\n")

    assert result.atomic is False
    assert result.fallback_reason == "EACCES"
    assert target.read_text(encoding="utf-8") == "new body\n"


def test_executable_bit_survives_edit(tmp_path: Path) -> None:
    script = tmp_path / "collect.sh"
    script.write_text("#!/bin/sh\necho old\n", encoding="utf-8")
    script.chmod(0o755)
    tool = _tool(tmp_path, "search_replace")

    tool.invoke({"path": "collect.sh", "old_string": "echo old", "new_string": "echo new"})

    assert "echo new" in script.read_text(encoding="utf-8")
    assert _file_mode(script) == 0o755


def test_symlink_writes_through_to_real_file(tmp_path: Path) -> None:
    real = tmp_path / "real.py"
    real.write_text(_ORIGINAL, encoding="utf-8")
    link = tmp_path / "link.py"
    link.symlink_to(real)

    result = write_workspace_text(link, "linked body\n")

    assert link.is_symlink()
    assert result.path == real
    assert real.read_text(encoding="utf-8") == "linked body\n"


def test_hardlink_write_keeps_link_relation(tmp_path: Path) -> None:
    primary = tmp_path / "a.py"
    primary.write_text(_ORIGINAL, encoding="utf-8")
    mirror = tmp_path / "b.py"
    os.link(primary, mirror)

    result = write_workspace_text(primary, "shared body\n")

    assert result.atomic is False
    assert result.fallback_reason == "hardlink"
    assert primary.stat().st_nlink == 2
    assert mirror.read_text(encoding="utf-8") == "shared body\n"


def test_write_file_creates_new_file_without_residue(tmp_path: Path) -> None:
    tool = _tool(tmp_path, "write_file")

    out = str(tool.invoke({"path": "pkg/new.py", "content": "x = 1\n"}))

    assert "已写入" in out
    assert (tmp_path / "pkg" / "new.py").read_text(encoding="utf-8") == "x = 1\n"
    assert _leftover_temps(tmp_path / "pkg") == []


def test_append_file_keeps_atomic_write(tmp_path: Path) -> None:
    target = tmp_path / "notes.md"
    target.write_text("head\n", encoding="utf-8")
    tool = _tool(tmp_path, "append_file")

    tool.invoke({"path": "notes.md", "content": "tail\n"})

    assert target.read_text(encoding="utf-8") == "head\ntail\n"
    assert _leftover_temps(tmp_path) == []


def test_sweep_removes_only_our_stale_temp_files(tmp_path: Path) -> None:
    stale = temp_sibling_path(tmp_path / "app.py")
    stale.write_text("junk", encoding="utf-8")
    old = time.time() - 7200
    os.utime(stale, (old, old))
    fresh = temp_sibling_path(tmp_path / "app.py")
    fresh.write_text("junk", encoding="utf-8")
    user_file = tmp_path / ".editor.tmp"
    user_file.write_text("user data", encoding="utf-8")
    os.utime(user_file, (old, old))

    removed = sweep_stale_temp_files(tmp_path)

    assert removed == 1
    assert not stale.exists()
    assert fresh.exists()
    assert user_file.read_text(encoding="utf-8") == "user data"
