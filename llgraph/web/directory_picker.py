"""本机目录选择对话框（Web 打开工作区）。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def pick_directory(*, initial_path: str = "") -> str | None:
    """
    唤起系统目录选择对话框。

    @param initial_path 初始目录
    @return 选中目录绝对路径；取消或不可用时 None
    """
    home = str(Path.home())
    initial = initial_path.strip() or home
    try:
        initial = str(Path(initial).expanduser().resolve())
    except OSError:
        initial = home

    if sys.platform == "darwin":
        return _pick_macos(initial)
    if sys.platform.startswith("linux"):
        return _pick_linux(initial)
    if sys.platform == "win32":
        return _pick_windows(initial)
    return None


def _pick_macos(initial: str) -> str | None:
    escaped = initial.replace("\\", "\\\\").replace('"', '\\"')
    script = f'''
try
    set defaultPath to POSIX file "{escaped}"
    set chosenFolder to choose folder with prompt "选择工作区目录" default location defaultPath
    return POSIX path of chosenFolder
on error number -128
    return ""
end try
'''
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    path = (result.stdout or "").strip()
    return path or None


def _pick_linux(initial: str) -> str | None:
    for cmd in (
        ["zenity", "--file-selection", "--directory", "--title=选择工作区目录"],
        ["kdialog", "--getexistingdirectory", initial, "--title", "选择工作区目录"],
    ):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode != 0:
            continue
        path = (result.stdout or "").strip()
        if path:
            return str(Path(path).expanduser().resolve())
    return None


def _pick_windows(initial: str) -> str | None:
    escaped = initial.replace("'", "''")
    script = f"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = '选择工作区目录'
$dialog.SelectedPath = '{escaped}'
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
    Write-Output $dialog.SelectedPath
}}
"""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    path = (result.stdout or "").strip()
    return path or None
