"""交互输入：readline 历史、退出识别、合并多行粘贴。"""

from __future__ import annotations

import atexit
import sys
import time
from pathlib import Path

from llgraph.ui.keys import is_exit_command

# 须在首次 input() 之前加载，macOS 上才能让 ↑↓ 翻阅历史
try:
    import readline as _readline_mod  # noqa: F401
except ImportError:
    _readline_mod = None

_PASTE_END_MARKERS = frozenset({"---", "END", "/end", "/done", "/send"})
_PASTE_PROMPT = "paste> "
_HISTORY_FILE = Path.home() / ".llgraph" / "readline_history"
_HISTORY_MAX = 500
_READLINE_READY = False


def _import_readline():
    """
    导入 readline（macOS/Homebrew 上可能仅有历史 API，无 readline.readline）。

    @return readline 模块或 None
    """
    try:
        import readline as rl_mod
    except ImportError:
        return None
    if not hasattr(rl_mod, "read_history_file"):
        return None
    return rl_mod


def init_input_history() -> None:
    """加载 readline 历史（TTY 下）。"""
    global _READLINE_READY
    if _READLINE_READY:
        return
    if not sys.stdin.isatty():
        return
    rl_mod = _import_readline()
    if rl_mod is None:
        return
    try:
        _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        if _HISTORY_FILE.is_file():
            rl_mod.read_history_file(str(_HISTORY_FILE))
        if hasattr(rl_mod, "set_history_length"):
            rl_mod.set_history_length(_HISTORY_MAX)
        if sys.platform == "darwin":
            if hasattr(rl_mod, "parse_and_bind"):
                rl_mod.parse_and_bind("bind ^I rl_complete")
                rl_mod.parse_and_bind("bind '\\e[A' history-search-backward")
                rl_mod.parse_and_bind("bind '\\e[B' history-search-forward")
        atexit.register(save_input_history)
        _READLINE_READY = True
    except Exception:
        pass


def save_input_history() -> None:
    """保存 readline 历史。"""
    if not _READLINE_READY:
        return
    rl_mod = _import_readline()
    if rl_mod is None:
        return
    try:
        rl_mod.write_history_file(str(_HISTORY_FILE))
    except Exception:
        pass


def _append_input_history(line: str) -> None:
    """
    将一行加入 readline 历史（供 ↑ 翻阅）。

    @param line 用户输入
    """
    if not line.strip() or not _READLINE_READY:
        return
    rl_mod = _import_readline()
    if rl_mod is None or not hasattr(rl_mod, "add_history"):
        return
    try:
        rl_mod.add_history(line)
    except Exception:
        pass


def _read_tty_line(prompt: str) -> str:
    """
    从 TTY 读一行（兼容无 readline.readline 的 libedit 构建）。

    @param prompt 提示符
    @return 去掉换行的输入
    @raises EOFError 输入结束
    """
    rl_mod = _import_readline()
    if rl_mod is not None and hasattr(rl_mod, "readline"):
        line = rl_mod.readline(prompt)
        if line == "":
            raise EOFError
        return line.rstrip("\r\n")
    try:
        return input(prompt).rstrip("\r\n")
    except EOFError:
        raise


def _stdin_ready(timeout: float = 0.0) -> bool:
    if not sys.stdin.isatty():
        return False
    if sys.platform == "win32":
        return False
    import select

    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    return bool(ready)


def _readline_strip() -> str:
    line = sys.stdin.readline()
    if line == "":
        raise EOFError
    return line.rstrip("\r\n")


def _read_paste_line() -> str:
    """
    粘贴模式读一行（TTY 下带 paste> 提示符）。

    @return 用户输入行
    @raises EOFError 输入结束
    @raises KeyboardInterrupt 用户取消粘贴
    """
    if sys.stdin.isatty():
        return _read_tty_line(_PASTE_PROMPT)
    return _readline_strip()


def _read_paste_block() -> str:
    """
    显式多行粘贴模式，单独一行 --- 或 END 结束。

    @return 合并后的文本；取消时返回空字符串
    """
    print(
        "多行粘贴模式：粘贴内容后任选一种结束\n"
        "  · 单独一行输入 ---\n"
        "  · 或连续两次回车\n"
        "  · Ctrl+C 取消并回到对话",
        flush=True,
    )
    lines: list[str] = []
    saw_empty = False
    while True:
        try:
            line = _read_paste_line()
        except KeyboardInterrupt:
            print("\n[已取消粘贴]", flush=True)
            return ""
        except EOFError:
            break
        stripped = line.strip()
        if stripped in _PASTE_END_MARKERS:
            break
        if not stripped:
            if saw_empty:
                break
            saw_empty = True
            continue
        saw_empty = False
        lines.append(line)
    return "\n".join(lines).strip()


def _drain_buffered_lines(timeout: float = 0.35) -> list[str]:
    """合并终端缓冲区中连续到达的行。"""
    extra: list[str] = []
    while _stdin_ready(timeout):
        try:
            extra.append(_readline_strip())
        except EOFError:
            break
        timeout = 0.05
    return extra


def read_interactive_user_message() -> str:
    """
    读取一条用户消息（含粘贴合并与 /paste）。

    @return 用户输入（可能多行）
    @raises EOFError 输入结束
    """
    prompt = "\n> "
    if sys.stdin.isatty():
        line = _read_tty_line(prompt)
    else:
        print(prompt, end="", flush=True)
        line = _readline_strip()

    if not line.strip():
        return ""

    if line.strip().lower() in ("/paste", "/p"):
        return _read_paste_block()

    merged = [line]
    merged.extend(_drain_buffered_lines())
    if len(merged) > 1:
        print(f"[已合并 {len(merged)} 行粘贴]", flush=True)
    return "\n".join(merged).strip()
