"""对话输出：按当前 UI（TUI / 终端）路由。"""

from __future__ import annotations


def write_dialog_block(text: str) -> None:
    """
    向当前 UI 写入多行文本。

    @param text 多行内容
    """
    from llgraph.ui.context import get_ui_app

    app = get_ui_app()
    if app is not None:
        app.write_chat_block(text.strip())
        return
    if text.strip():
        print(text.strip(), flush=True)


def write_dialog_line(text: str) -> None:
    """
    向当前 UI 写入单行。

    @param text 一行文本
    """
    from llgraph.ui.context import get_ui_app

    app = get_ui_app()
    if app is not None:
        app.write_chat_line(text, dim=False)
        return
    if text.strip():
        print(text, flush=True)
