"""文件路径权限：工作区边界与 read_file 可读白名单。"""

from __future__ import annotations

from pathlib import Path

from llgraph.core.agent_config import USER_LLGRAPH_HOME
from llgraph.session.user_storage import user_rules_dir, user_skills_dir
from llgraph.sandbox.policy import SandboxPolicy


def resolve_workspace_path(
    workspace: Path,
    relative_path: str,
    *,
    sandbox: SandboxPolicy | None = None,
    for_write: bool = False,
) -> Path:
    """
    解析相对工作区的路径，禁止跳出根目录（写操作与 shell cwd 使用）。

    @param workspace 工作区根
    @param relative_path 相对路径，空或 "." 表示工作区根
    @param sandbox 可选沙箱策略
    @param for_write 为 True 时额外做沙箱写权限校验
    @return 绝对路径
    """
    root = workspace.expanduser().resolve()
    raw = (relative_path or ".").strip()
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"路径超出工作区范围: {relative_path}") from exc
    if for_write and sandbox is not None and sandbox.enabled:
        denial = sandbox.check_write(candidate)
        if denial:
            raise PermissionError(sandbox.format_denial(denial))
    return candidate


def resolve_read_path(
    workspace: Path,
    path_str: str,
    *,
    sandbox: SandboxPolicy | None = None,
) -> Path:
    """
    read_file 路径解析：工作区相对路径，或 ~/.llgraph 下技能/规则等目录内绝对路径。

    @param workspace 工作区根
    @param path_str 目录中给出的路径句柄
    @param sandbox 可选沙箱策略；启用时仅允许 sandbox.json 声明的路径
    @return 可读绝对路径
    """
    raw = (path_str or "").strip()
    if not raw:
        raise ValueError("path 不能为空")

    expanded = Path(raw).expanduser()
    root = workspace.expanduser().resolve()

    if sandbox is not None and sandbox.enabled:
        if not expanded.is_absolute():
            target = (root / raw).resolve()
        else:
            target = expanded.resolve()
        denial = sandbox.check_read(target)
        if denial:
            raise ValueError(sandbox.format_denial(denial))
        return target

    if not expanded.is_absolute():
        return resolve_workspace_path(root, raw)

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
