"""``llgraph web`` 子命令。"""

from __future__ import annotations

import argparse
import os
import socket
import sys


def _lan_urls(port: int) -> list[str]:
    """本机非 loopback IPv4 地址（供局域网访问提示）。"""
    urls: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127."):
                urls.append(f"http://{ip}:{port}/console")
    except OSError:
        pass
    # 去重保序
    seen: set[str] = set()
    ordered: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered


def main(argv: list[str] | None = None) -> None:
    """
    启动 bundled Web Console（本地 UI + 内部 HTTP 传输）。

    对外集成请使用 ``llgraph.console.Console`` Python API 或终端 CLI。

    @param argv 命令行参数（不含 ``web``）
    """
    try:
        import uvicorn
    except ImportError as exc:
        print(
            "错误: 未安装 web 可选依赖。\n"
            "  推荐: ./scripts/setup.sh web\n"
            "  或:   uv sync --extra web  /  pip install -e '.[web]'",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    parser = argparse.ArgumentParser(
        description="llgraph Web Console（本地 UI，集成请用 llgraph.console 库）",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("LLGRAPH_WEB_HOST", "127.0.0.1"),
        help="绑定地址（默认 127.0.0.1 仅本机；局域网请 0.0.0.0 或 LLGRAPH_WEB_HOST）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("LLGRAPH_WEB_PORT", "8765")),
        help="端口",
    )
    parser.add_argument(
        "--static",
        default="",
        help="前端静态目录（默认 <repo>/web-ui/dist）",
    )
    args = parser.parse_args(argv)

    if args.static:
        os.environ["LLGRAPH_WEB_STATIC"] = args.static

    print(f"llgraph Web Console: http://{args.host}:{args.port}/console")
    if args.host in ("0.0.0.0", "::"):
        print(f"  本机: http://127.0.0.1:{args.port}/console")
        for url in _lan_urls(args.port):
            print(f"  局域网: {url}")
    elif args.host == "127.0.0.1":
        print("  提示: 局域网访问请 llgraph web --host 0.0.0.0")

    uvicorn.run(
        "llgraph.web.server.app:app",
        host=args.host,
        port=args.port,
        reload=False,
        timeout_graceful_shutdown=5,
    )
