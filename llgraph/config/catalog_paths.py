"""Skill / Rule 目录展示路径（工作区相对或个人绝对路径）。"""

from __future__ import annotations

from pathlib import Path

from llgraph.core.agent_config import USER_LLGRAPH_HOME
from llgraph.session.user_storage import user_rules_dir, user_skills_dir


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


def resolve_catalog_read_path(workspace: Path, path_str: str) -> Path:
    """
    read_file 路径解析：工作区相对路径，或 ~/.llgraph 下技能/规则等目录内绝对路径。

    @param workspace 工作区根
    @param path_str 目录中给出的路径句柄
    @return 可读绝对路径
    """
    raw = (path_str or "").strip()
    if not raw:
        raise ValueError("path 不能为空")

    expanded = Path(raw).expanduser()
    root = workspace.expanduser().resolve()

    if not expanded.is_absolute():
        candidate = (root / raw).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"路径超出工作区范围: {path_str}") from exc
        return candidate

    resolved = expanded.resolve()
    allowed_roots = (
        root,
        USER_LLGRAPH_HOME.resolve(),
        user_skills_dir().resolve(),
        user_rules_dir().resolve(),
    )
    for base in allowed_roots:
        if not base.exists():
            continue
        try:
            resolved.relative_to(base)
            return resolved
        except ValueError:
            continue
    raise ValueError(
        f"路径超出允许范围: {path_str}（仅支持工作区相对路径或 ~/.llgraph 下技能/规则）"
    )
