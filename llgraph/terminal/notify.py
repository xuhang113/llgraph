"""终端状态通知（index-watch 等后台任务）。"""

from __future__ import annotations

from llgraph.terminal.style import sty


def notify(tag: str, message: str) -> None:
    """
    输出带标签的状态通知。

    @param tag 标签
    @param message 说明
    """
    style = "ok" if "已启动" in message or "成功" in message else "value"
    print(f"{sty(f'[{tag}]', 'label')} {sty(message, style)}", flush=True)
