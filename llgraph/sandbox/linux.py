"""Linux bubblewrap 沙箱命令构建。"""

from __future__ import annotations

from pathlib import Path

from llgraph.sandbox.policy import SandboxPolicy


def build_bwrap_command(
    policy: SandboxPolicy,
    *,
    command: str,
    cwd: Path,
) -> list[str]:
    """
    构建 bwrap 命令行。

    @param policy 沙箱策略
    @param command shell 命令
    @param cwd 工作目录
    @return argv 列表
    """
    argv = ["bwrap", "--ro-bind", "/", "/"]

    bound: set[str] = set()

    def bind_path(path: Path, *, readwrite: bool) -> None:
        resolved = path.expanduser().resolve()
        key = str(resolved)
        if key in bound or not resolved.exists():
            return
        bound.add(key)
        if readwrite:
            argv.extend(["--bind", key, key])
        else:
            argv.extend(["--ro-bind", key, key])

    for root in policy.readonly_roots:
        bind_path(root, readwrite=False)

    if policy.mode == "workspace_readwrite":
        for root in policy.readwrite_roots:
            bind_path(root, readwrite=True)

    if policy.allow_tmp_write:
        argv.extend(["--tmpfs", "/tmp"])

    argv.extend(["--dev", "/dev", "--proc", "/proc"])

    if policy.network == "deny":
        argv.extend(["--unshare-net"])

    argv.extend([
        "--chdir", str(cwd),
        "--",
        "/bin/sh", "-c", command,
    ])
    return argv
