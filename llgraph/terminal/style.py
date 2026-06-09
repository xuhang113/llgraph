"""终端追踪行样式（ANSI 语义，供 trace_display 构建文本）。"""

from __future__ import annotations

import os
import sys

_RESET = "\033[0m"

STYLES: dict[str, str] = {
    "reset": "0",
    "bold": "1",
    "dim": "2",
    "title": "1;35",
    "brand": "1;36",
    "label": "36",
    "value": "97",
    "path": "34",
    "number": "33",
    "ok": "32",
    "warn": "33",
    "err": "31",
    "hint": "90",
    "cmd": "36",
    "accent": "35",
    "prompt": "1;32",
    "tag": "90",
}

_TRACE_L1 = "  "
_TRACE_L2 = "    "
_TRACE_L3 = "      "


def indent_line(level: int = 1) -> str:
    """
    层级缩进前缀。

    @param level 层级
    @return 空格前缀
    """
    if level <= 0:
        return ""
    return "  " * level


def _is_interactive_tty() -> bool:
    """
    stdin/stdout 均为 TTY 时视为交互终端。

    @return 是否交互 TTY
    """
    stdin_tty = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
    stdout_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    return stdin_tty and stdout_tty


def color_enabled() -> bool:
    """
    是否启用 ANSI 着色。

    LLGRAPH_COLOR=1 可强制着色；LLGRAPH_COLOR=0 强制关闭。
    交互终端默认着色（与 prompt-toolkit 斜杠补全一致），NO_COLOR 仅影响非交互输出。

    @return 是否着色
    """
    llgraph_color = os.environ.get("LLGRAPH_COLOR", "").strip().lower()
    if llgraph_color in ("1", "true", "yes", "on"):
        return True
    if llgraph_color in ("0", "false", "no", "off"):
        return False
    if _is_interactive_tty():
        return True
    if os.environ.get("NO_COLOR", "").strip():
        return False
    if os.environ.get("FORCE_COLOR", "").strip():
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def sty(text: str, style: str = "value") -> str:
    """
    套用语义样式。

    @param text 原文
    @param style STYLES 键名
    @return 带 ANSI 的文本
    """
    if not text or not color_enabled():
        return text
    code = STYLES.get(style, style)
    return f"\033[{code}m{text}{_RESET}"


def sty_sgr(text: str, code: str) -> str:
    """
    使用 SGR 码着色。

    @param text 原文
    @param code SGR 或 STYLES 键
    @return 着色文本
    """
    return sty(text, code)
