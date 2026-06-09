#!/usr/bin/env python3
"""静态 + 动态 import 检查：捕获缺失 import / 未定义名称（F821）。"""

from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys


def _run_ruff() -> int:
    cmd = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "llgraph",
        "scripts",
        "--select",
        "F821,F823",
    ]
    print("$", " ".join(cmd))
    return subprocess.call(cmd)


def _import_all_modules() -> int:
    pkg = importlib.import_module("llgraph")
    failed: list[tuple[str, str]] = []
    ok = 0
    for mod in pkgutil.walk_packages(pkg.__path__, prefix="llgraph."):
        try:
            importlib.import_module(mod.name)
            ok += 1
        except Exception as exc:
            failed.append((mod.name, f"{type(exc).__name__}: {exc}"))
    print(f"import_all: ok={ok} failed={len(failed)}")
    for name, msg in sorted(failed):
        print(f"  FAIL {name}: {msg}")
    return 1 if failed else 0


def main() -> int:
    code = _run_ruff()
    if code != 0:
        return code
    return _import_all_modules()


if __name__ == "__main__":
    raise SystemExit(main())
