"""交互输入：readline 历史、退出识别、合并多行粘贴。"""

from __future__ import annotations

import atexit
import sys
import time
from pathlib import Path

from llgraph.ui.output import emit, emit_hint

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
    emit(
        "多行粘贴模式：粘贴内容后任选一种结束\n"
        "  · 单独一行输入 ---\n"
        "  · 或连续两次回车\n"
        "  · Ctrl+C 取消并回到对话",
        colorize=True,
    )
    lines: list[str] = []
    saw_empty = False
    while True:
        try:
            line = _read_paste_line()
        except KeyboardInterrupt:
            emit("\n[已取消粘贴]", colorize=True)
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


def read_interactive_user_message(workspace: Path | None = None) -> str:
    """
    读取一条用户消息（含粘贴合并、/paste、斜杠补全）。

    @param workspace 工作区根（斜杠补全 Skills/Commands）
    @return 用户输入（可能多行）
    @raises EOFError 输入结束
    """
    prompt = "\n> "
    if sys.stdin.isatty() and workspace is not None:
        from llgraph.terminal.slash_complete import prompt_toolkit_available

        if prompt_toolkit_available():
            line = _read_tty_line_with_slash_complete(workspace, prompt)
        else:
            line = _read_tty_line(prompt)
    elif sys.stdin.isatty():
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
        emit_hint(f"[已合并 {len(merged)} 行粘贴]")
    return "\n".join(merged).strip()


def _read_tty_line_with_slash_complete(workspace: Path, prompt: str) -> str:
    """
    prompt_toolkit 输入：/ 触发 Skills/Commands 动态补全。

    @param workspace 工作区根
    @param prompt 提示符
    @return 用户输入行
    @raises EOFError 输入结束
    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.filters import has_completions
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.shortcuts import CompleteStyle

    from llgraph.terminal.slash_complete import build_slash_completer
    from llgraph.terminal.slash_completion_menu import patch_slash_completion_menu_full_width
    from llgraph.terminal.slash_prompt_theme import SLASH_COMPLETION_STYLE

    bindings = KeyBindings()

    @bindings.add("down", filter=has_completions, eager=True)
    @bindings.add("c-n", filter=has_completions, eager=True)
    def _complete_down(event) -> None:
        buff = event.current_buffer
        state = buff.complete_state
        if state is None or not state.completions:
            return
        if state.complete_index is None:
            buff.go_to_completion(0)
        else:
            buff.complete_next()

    @bindings.add("up", filter=has_completions, eager=True)
    @bindings.add("c-p", filter=has_completions, eager=True)
    def _complete_up(event) -> None:
        buff = event.current_buffer
        state = buff.complete_state
        if state is None or not state.completions:
            return
        if state.complete_index is None:
            buff.go_to_completion(len(state.completions) - 1)
        else:
            buff.complete_previous()

    @bindings.add("enter", filter=has_completions)
    def _enter_apply_completion(event) -> None:
        buff = event.current_buffer
        state = buff.complete_state
        # 仅当用户用 ↑↓ 选中某项时 Enter 才插入；未选中则关闭菜单并提交当前输入
        if state is not None and state.complete_index is not None:
            completion = state.current_completion
            if completion is not None:
                buff.apply_completion(completion)
            buff.cancel_completion()
            return
        buff.cancel_completion()
        buff.validate_and_handle()

    session = PromptSession(
        completer=build_slash_completer(workspace),
        complete_while_typing=True,
        complete_in_thread=False,
        complete_style=CompleteStyle.COLUMN,
        style=SLASH_COMPLETION_STYLE,
        include_default_pygments_style=False,
        key_bindings=bindings,
        reserve_space_for_menu=10,
    )
    patch_slash_completion_menu_full_width(session)

    try:
        line = session.prompt(ANSI(prompt))
    except KeyboardInterrupt:
        raise
    except EOFError:
        raise EOFError from None
    return line.rstrip("\r\n")
