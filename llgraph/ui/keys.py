"""TUI 快捷键与退出口令（唯一维护点）。"""

from __future__ import annotations

EXIT_ALIASES = frozenset({"exit", "quit", "q", "/exit", "/quit"})

MSG_GOODBYE = "再见。"
MSG_INTERRUPT_EXIT = "已中断（Ctrl+C），退出。"
MSG_MENU_CANCELLED = "已取消。"

HELP_SHORTCUT_LINES: tuple[str, ...] = (
    "  exit, quit, q      退出",
    "  TUI: Ctrl+C / Ctrl+Q 退出 · Ctrl+O 步骤侧栏",
    "  terminal: ↑↓ 历史 · /paste 多行粘贴",
    "  /help /trace /paste  元命令",
    "  --ui tui            Textual TUI（默认 terminal；或 LLGRAPH_UI）",
)

BANNER_SHORTCUT_HINT = "Ctrl+Q 退出 · /help"


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
