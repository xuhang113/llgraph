"""Skill / Rule 目录展示路径（工作区相对或个人绝对路径）。"""

from __future__ import annotations

from pathlib import Path

from llgraph.permissions.paths import resolve_read_path

from llgraph.sandbox.policy import SandboxPolicy


def format_catalog_path(workspace: Path, path: Path, scope: str) -> str:
    """
    供 read_file 使用的路径字符串。

    @param workspace 工作区根
    @param path 源文件或目录
    @param scope workspace | user
    @return 路径字符串
    """
    if scope == "user":
        return path.expanduser().resolve().as_posix()
    root = workspace.expanduser().resolve()
    try:
        return path.expanduser().resolve().relative_to(root).as_posix()
    except ValueError:
        return path.expanduser().resolve().as_posix()


def scope_label(scope: str) -> str:
    """
    来源标签。

    @param scope workspace | user
    @return 中文标签
    """
    if scope == "user":
        return "个人"
    return "项目"


def resolve_catalog_read_path(
    workspace: Path,
    path_str: str,
    *,
    sandbox: SandboxPolicy | None = None,
) -> Path:
    """
    read_file 路径解析：工作区相对路径，或 ~/.llgraph 下技能/规则等目录内绝对路径。

    @param workspace 工作区根
    @param path_str 目录中给出的路径句柄
    @param sandbox 可选沙箱策略
    @return 可读绝对路径
    """
    return resolve_read_path(workspace, path_str, sandbox=sandbox)
