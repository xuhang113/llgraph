"""运维类提示：默认不占用对话区，需时用 LLGRAPH_VERBOSE_CONTEXT=1 打开。"""

from __future__ import annotations

import os
import sys


def ops_notice_visible() -> bool:
    """
    是否在终端展示裁剪/压缩/修链等运维提示。

    @return 是否展示
    """
    raw = os.environ.get("LLGRAPH_VERBOSE_CONTEXT", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def ops_notice(message: str) -> None:
    """
    输出运维提示（默认静默，对齐 Cursor 不在对话区刷内部日志）。

    @param message 单行说明
    """
    if not ops_notice_visible():
        return
    text = message.strip()
    if not text:
        return
    print(f"[llgraph] {text}", file=sys.stderr, flush=True)
