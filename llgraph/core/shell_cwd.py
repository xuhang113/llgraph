"""解析 shell 命令开头的 cd，供会话工作目录连续（对齐 Claude Code Bash）。

每条命令都是新进程，`cd foo && pytest` 的 cwd 默认不会带到下一轮。
剥掉开头的 cd 在父进程解析并记住，下一轮默认落在同一目录，少一轮 LLM。
"""

from __future__ import annotations

import re
from pathlib import Path

from llgraph.core.workspace import WorkspaceContext

_CD_RE = re.compile(
    r"^\s*cd\s+(?P<target>(?:'[^']+'|\"[^\"]+\"|[^\s;&|]+))\s*"
    r"(?:(?P<sep>&&|;)\s*(?P<rest>.*))?$",
    re.DOTALL,
)


def unquote_cd_target(raw: str) -> str:
    """去掉 cd 目标上的一层引号。"""
    text = (raw or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def peel_leading_cd(command: str) -> tuple[str | None, str]:
    """
    若命令以 `cd TARGET` 开头，剥掉这一段。

    不处理 `cd -` / `cd ~` / 子 shell `(cd …)`，避免跳出工作区或改变语义。

    @param command 原始命令
    @return (目标目录或 None, 剩余命令；无 cd 时剩余为原文)
    """
    text = command or ""
    match = _CD_RE.match(text)
    if not match:
        return None, text
    target = unquote_cd_target(match.group("target"))
    if not target or target == "-" or target.startswith("~"):
        return None, text
    rest = (match.group("rest") or "").strip()
    return target, rest


def peel_all_leading_cd(command: str) -> tuple[list[str], str]:
    """
    连续剥掉开头的 `cd a && cd b && cmd`。

    @param command 原始命令
    @return (相对当前目录的 cd 链, 剩余命令)
    """
    hops: list[str] = []
    rest = command or ""
    while True:
        target, rest = peel_leading_cd(rest)
        if target is None:
            break
        hops.append(target)
        if not rest:
            break
    return hops, rest


def resolve_cwd_rel(
    ctx: WorkspaceContext,
    current_rel: str,
    target: str,
) -> tuple[str | None, str]:
    """
    把 cd 目标解析成工作区内相对路径。

    @param ctx 工作区
    @param current_rel 当前相对 cwd
    @param target cd 目标（相对 current 或绝对）
    @return (新相对路径, 错误)；成功时错误为空
    """
    try:
        base = ctx.resolve_path(current_rel or ".")
    except (ValueError, RuntimeError, PermissionError) as exc:
        return None, f"工作目录无效: {exc}"
    raw = Path(target)
    try:
        abs_path = raw.resolve() if raw.is_absolute() else (base / target).resolve()
    except (OSError, RuntimeError) as exc:
        return None, f"工作目录无效: {exc}"
    try:
        rel = abs_path.relative_to(ctx.root).as_posix()
    except ValueError:
        return None, f"工作目录超出工作区: {target}"
    if not abs_path.is_dir():
        return None, f"工作目录不存在或不是目录: {rel or target}"
    return (rel if rel != "." else ".") or ".", ""


def apply_cd_hops(
    ctx: WorkspaceContext,
    current_rel: str,
    hops: list[str],
) -> tuple[str | None, str]:
    """
    依次应用 cd 链。

    @param ctx 工作区
    @param current_rel 起始相对 cwd
    @param hops cd 目标列表
    @return (最终相对路径, 错误)
    """
    cwd = current_rel or "."
    for hop in hops:
        nxt, err = resolve_cwd_rel(ctx, cwd, hop)
        if err:
            return None, err
        cwd = nxt or "."
    return cwd, ""
