"""工具开始跑了：单次工具调用的「起步」通知。

trace 的步骤一律在工具**跑完**后才登记（那时才有耗时与输出），所以编辑器里一个
长命令（跑测试）期间看不到任何动静。这里补的就是那条缺失的「开始」事件：
``ToolNode`` 每执行一次工具调用之前叫一声，入口（ACP）据此把
``tool_call`` 从 pending 推进到 in_progress。

观察者放在 ContextVar 上，理由与 `permissions/approval.py` 的闸门、
`core/editor_fs.py` 的文件来源一样：工具在 LangGraph 的线程池里跑，
langchain 提交任务时复制调用方 context，全局变量则会在多会话同进程时串台。

没有入口登记观察者时这里一律是空操作——CLI / Web Console 的行为不变。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Callable, Iterator

ToolStartObserver = Callable[[str, str], None]
"""``(tool_call_id, tool_name) -> None``；在工具真正开跑之前被调用。"""

_observer: ContextVar[ToolStartObserver | None] = ContextVar(
    "llgraph_tool_start_observer", default=None
)


def set_tool_start_observer(observer: ToolStartObserver | None) -> Token:
    """
    登记工具起步观察者。

    @param observer 观察者；None 表示不通知
    @return ContextVar token，交给 ``reset_tool_start_observer``
    """
    return _observer.set(observer)


def reset_tool_start_observer(token: Token) -> None:
    """
    还原上一层观察者。

    @param token ``set_tool_start_observer`` 的返回值
    """
    _observer.reset(token)


@contextmanager
def use_tool_start_observer(observer: ToolStartObserver | None) -> Iterator[None]:
    """
    在一段执行期间登记观察者（一轮 invoke 外面套一层）。

    @param observer 观察者；None 时等于什么都不做
    """
    token = set_tool_start_observer(observer)
    try:
        yield
    finally:
        reset_tool_start_observer(token)


def current_tool_start_observer() -> ToolStartObserver | None:
    """@return 当前观察者；无则 None"""
    return _observer.get()


def notify_tool_started(tool_call_id: str, tool_name: str) -> None:
    """
    通知「这次工具调用开始跑了」。

    观察者自己抛异常一律吞掉：它只是给界面加进度，不能把工具执行带崩。

    @param tool_call_id 工具调用 ID（模型给的那个）
    @param tool_name 工具名
    """
    observer = _observer.get()
    if observer is None:
        return
    cid = str(tool_call_id or "").strip()
    if not cid:
        return
    try:
        observer(cid, str(tool_name or "").strip())
    except Exception:  # noqa: BLE001 - 进度通知失败不能升级成工具崩溃
        return
