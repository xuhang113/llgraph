"""终端快捷键与退出口令（唯一维护点）。"""

from __future__ import annotations

EXIT_ALIASES = frozenset({"exit", "quit", "q", "/exit", "/quit"})

MSG_GOODBYE = "再见。"
MSG_INTERRUPT_EXIT = "已中断（Ctrl+C），退出。"
MSG_MENU_CANCELLED = "已取消。"

HELP_SHORTCUT_LINES: tuple[str, ...] = (
    "  exit, quit, q      退出",
    "  Ctrl+C             中断当前轮并退出",
    "  ↑↓ 历史 · /paste   多行粘贴",
    "  /help /trace /paste  元命令",
)


def is_exit_command(text: str) -> bool:
    """
    是否为退出指令（支持 exit;）。

    @param text 用户输入
    @return 是否退出
    """
    raw = text.strip().lower().rstrip(";")
    if raw in EXIT_ALIASES:
        return True
    parts = [p.strip().lower().rstrip(";") for p in raw.split(";") if p.strip()]
    return bool(parts) and all(p in EXIT_ALIASES for p in parts)
