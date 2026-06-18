"""llgraph search 子命令（并行 / 语义检索调试）。"""

import argparse
import sys
from pathlib import Path

from llgraph.code_index.parallel_search import search_parallel
from llgraph.code_index.search import search_semantic
from llgraph.config.config import load_llgraph_env
from llgraph.config.logging_settings import level_name, setup_search_logging


def main(argv: list[str] | None = None) -> None:
    """并行 / 语义检索 CLI。"""
    parser = argparse.ArgumentParser(prog="llgraph search")
    parser.add_argument("query", help="检索问题或关键词")
    parser.add_argument("-C", "--workspace", default=".", metavar="DIR")
    parser.add_argument(
        "--mode",
        choices=["parallel", "hybrid", "semantic"],
        default="parallel",
        help="parallel（默认）= 字面量 grep + 向量；hybrid 为 parallel 别名；semantic = 纯向量",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--path", default=".", metavar="PREFIX")
    parser.add_argument(
        "--log-level",
        default=None,
        metavar="LEVEL",
        help="向量检索日志：debug|info|warning（默认 warning；INFO 以上会写 search.log）",
    )
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.is_dir():
        print(f"错误: 不是目录: {workspace}", file=sys.stderr)
        sys.exit(1)

    load_llgraph_env()
    effective = setup_search_logging(workspace, cli_override=args.log_level)
    if args.log_level:
        print(f"[llgraph] 向量检索日志级别: {level_name(effective)}", file=sys.stderr)

    if args.mode == "semantic":
        print(
            search_semantic(
                workspace,
                args.query,
                top_k=args.top_k,
                path_prefix=args.path,
                source="cli",
                tool="search_code_semantic",
            )
        )
    else:
        print(
            search_parallel(
                workspace,
                args.query,
                top_k=args.top_k,
                path_prefix=args.path,
                source="cli",
                tool="search_code_parallel",
            )
        )
