"""TUI 全局上下文（供 Agent 线程与工具回调使用）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llgraph.ui.app import LlgraphApp

_CURRENT_APP: LlgraphApp | None = None


def set_ui_app(app: LlgraphApp | None) -> None:
    """
    注册当前 Textual App。

    @param app 运行中的 App；退出时传 None
    """
    global _CURRENT_APP
    _CURRENT_APP = app


def get_ui_app() -> LlgraphApp | None:
    """
    获取当前 App。

    @return App 或 None
    """
    return _CURRENT_APP


def ui_log_line(text: str) -> None:
    """
    向主对话区追加一行（无 App 时忽略）。

    @param text 文本
    """
    app = _CURRENT_APP
    if app is not None:
        app.write_chat_line(text, dim=False)


def ui_notify(tag: str, message: str) -> None:
    """
    状态通知（index-watch 等后台任务）。

    @param tag 标签
    @param message 说明
    """
    app = _CURRENT_APP
    if app is not None:
        app.log_line(f"[{tag}] {message}")
        return
    from llgraph.ui.style import sty

    style = "ok" if "已启动" in message or "成功" in message else "value"
    print(f"{sty(f'[{tag}]', 'label')} {sty(message, style)}", flush=True)
