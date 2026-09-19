"""``llgraph acp`` 子命令：在 stdio 上提供 Agent Client Protocol 服务。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    """
    启动 ACP 服务端（由编辑器以子进程方式拉起，不给人直接敲）。

    @param argv 命令行参数（不含 ``acp``）
    """
    parser = argparse.ArgumentParser(
        prog="llgraph acp",
        description="Agent Client Protocol 服务端（stdio；供 Zed / Neovim 等编辑器接入）",
    )
    parser.add_argument(
        "-C",
        "--workspace",
        default=None,
        metavar="DIR",
        help="默认工作区；编辑器在 session/new 里给了 cwd 时以 cwd 为准",
    )
    parser.add_argument(
        "-w",
        "--write",
        action="store_true",
        help="写入与 shell 不再逐次确认（默认每次都在编辑器里弹授权框）",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="完全只读：不注册写工具，也不会弹授权框",
    )
    args = parser.parse_args(argv)

    workspace: Path | None = None
    if args.workspace:
        workspace = Path(args.workspace).expanduser().resolve()
        if not workspace.is_dir():
            print(f"错误: 工作区不是有效目录: {workspace}", file=sys.stderr)
            raise SystemExit(1)
    else:
        cwd = Path(os.getcwd())
        if cwd.is_dir():
            workspace = cwd

    from llgraph.config.config import load_llgraph_env

    load_llgraph_env()

    from llgraph.core.llm import verify_model_credentials

    try:
        verify_model_credentials(workspace)
    except RuntimeError as exc:
        # 编辑器只看得到 stderr：凭据没配要在这里说清楚，不能等首轮提问才报
        print(f"配置错误: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    from llgraph.editor.acp.server import serve_stdio

    allow_write = not args.read_only
    ask_permission = allow_write and not args.write
    if not allow_write:
        mode = "只读"
    elif ask_permission:
        mode = "可写，每次写入/执行需在编辑器里确认"
    else:
        mode = "可写，不逐次确认"
    print(f"[llgraph] ACP 服务已就绪（stdio；{mode}）", file=sys.stderr, flush=True)
    try:
        serve_stdio(
            allow_write=allow_write,
            ask_permission=ask_permission,
            default_workspace=workspace,
        )
    except KeyboardInterrupt:
        pass
