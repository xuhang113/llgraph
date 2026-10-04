"""单次工具调用期间的进度通知：「开始跑了」「改了哪份文件」「挂在哪个终端上」。

trace 的步骤一律在工具**跑完**后才登记（那时才有耗时与输出），而步骤里留下的只是
工具返回的那段文本。三件事因此在链路上没有位置：

- **起步**：``ToolNode`` 每执行一次工具调用之前叫一声，入口（ACP）据此把
  ``tool_call`` 从 pending 推进到 in_progress，长命令期间编辑器里才有动静。
- **编辑**：写工具落盘那一刻的改前 / 改后全文，跑完就没了（步骤里只剩一句
  「已写入 x 字符」）。入口拿它渲染 diff，编辑器里才看得到这一刀改了什么。
- **终端**：命令交给编辑器终端跑的那一刻（``core/shell_terminal.py``）报一次
  终端 id，入口把它挂到这条 ``tool_call`` 上，输出才能边跑边进编辑器；
  等跑完再报就没有「实时」可言了。
- **计划**：``todo_write`` 落盘那一刻的整张清单。它和上面三件不一样——不属于某一次
  工具调用，而是会话级的一张表；入口（ACP）把它发成编辑器里的计划清单，
  而工具返回的那段文本只给模型看。

观察者放在 ContextVar 上，理由与 `permissions/approval.py` 的闸门、
`core/editor_fs.py` 的文件来源一样：工具在 LangGraph 的线程池里跑，
langchain 提交任务时复制调用方 context，全局变量则会在多会话同进程时串台。

没有入口登记观察者时这里一律是空操作——CLI / Web Console 的行为不变。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
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


@dataclass
class ToolEdit:
    """一次工具调用落在某份文件上的改动（改前 / 改后全文）。"""

    tool_call_id: str
    path: str
    old_text: str
    new_text: str


ToolEditObserver = Callable[[ToolEdit], None]
"""``(ToolEdit) -> None``；写工具落地之后被调用。"""

_edit_observer: ContextVar[ToolEditObserver | None] = ContextVar(
    "llgraph_tool_edit_observer", default=None
)

_current_call_id: ContextVar[str] = ContextVar(
    "llgraph_current_tool_call_id", default=""
)


def set_tool_edit_observer(observer: ToolEditObserver | None) -> Token:
    """
    登记工具编辑观察者。

    @param observer 观察者；None 表示不通知
    @return ContextVar token，交给 ``reset_tool_edit_observer``
    """
    return _edit_observer.set(observer)


def reset_tool_edit_observer(token: Token) -> None:
    """
    还原上一层观察者。

    @param token ``set_tool_edit_observer`` 的返回值
    """
    _edit_observer.reset(token)


@contextmanager
def use_tool_edit_observer(observer: ToolEditObserver | None) -> Iterator[None]:
    """
    在一段执行期间登记观察者（一轮 invoke 外面套一层）。

    @param observer 观察者；None 时等于什么都不做
    """
    token = set_tool_edit_observer(observer)
    try:
        yield
    finally:
        reset_tool_edit_observer(token)


def current_tool_edit_observer() -> ToolEditObserver | None:
    """@return 当前观察者；无则 None"""
    return _edit_observer.get()


@contextmanager
def use_current_tool_call(tool_call_id: str) -> Iterator[None]:
    """
    标记「当前正在跑的是哪次工具调用」（由 ToolNode 的调用包装登记）。

    写工具自己拿不到 tool_call_id（它只收到参数），可这条 id 正是入口把 diff
    挂回那一行的依据；并行写两份文件时少了它就分不清哪份属于哪次调用。

    @param tool_call_id 工具调用 ID（模型给的那个）
    """
    token = _current_call_id.set(str(tool_call_id or "").strip())
    try:
        yield
    finally:
        _current_call_id.reset(token)


def current_tool_call_id() -> str:
    """@return 当前正在跑的工具调用 ID；不在工具里时为空串"""
    return _current_call_id.get()


def notify_file_edited(path: str, old_text: str, new_text: str) -> None:
    """
    通知「这次工具调用把这份文件改成了这样」（落地之后才叫）。

    正文没变的写入不报：编辑器里画一个空 diff 只是噪声。
    认不出归属的调用也不报——入口按 tool_call_id 认行，没有 id 就挂不上去。

    @param path 工作区相对路径（入口负责补成绝对路径）
    @param old_text 改前全文；新建文件时为空串
    @param new_text 改后全文
    """
    observer = _edit_observer.get()
    if observer is None:
        return
    rel = str(path or "").strip()
    cid = _current_call_id.get()
    if not rel or not cid or old_text == new_text:
        return
    try:
        observer(ToolEdit(cid, rel, old_text, new_text))
    except Exception:  # noqa: BLE001 - 进度通知失败不能升级成工具崩溃
        return


ToolTerminalObserver = Callable[[str, str], None]
"""``(tool_call_id, terminal_id) -> None``；编辑器终端建好之后被调用。"""

_terminal_observer: ContextVar[ToolTerminalObserver | None] = ContextVar(
    "llgraph_tool_terminal_observer", default=None
)


def set_tool_terminal_observer(observer: ToolTerminalObserver | None) -> Token:
    """
    登记工具终端观察者。

    @param observer 观察者；None 表示不通知
    @return ContextVar token，交给 ``reset_tool_terminal_observer``
    """
    return _terminal_observer.set(observer)


def reset_tool_terminal_observer(token: Token) -> None:
    """
    还原上一层观察者。

    @param token ``set_tool_terminal_observer`` 的返回值
    """
    _terminal_observer.reset(token)


@contextmanager
def use_tool_terminal_observer(
    observer: ToolTerminalObserver | None,
) -> Iterator[None]:
    """
    在一段执行期间登记观察者（一轮 invoke 外面套一层）。

    @param observer 观察者；None 时等于什么都不做
    """
    token = set_tool_terminal_observer(observer)
    try:
        yield
    finally:
        reset_tool_terminal_observer(token)


def current_tool_terminal_observer() -> ToolTerminalObserver | None:
    """@return 当前观察者；无则 None"""
    return _terminal_observer.get()


def notify_terminal_created(terminal_id: str) -> None:
    """
    通知「这次工具调用的命令跑在这个编辑器终端里」。

    认不出归属的不报：入口按 tool_call_id 认行，没有 id 就挂不上去。

    @param terminal_id 编辑器给的终端 id
    """
    observer = _terminal_observer.get()
    if observer is None:
        return
    tid = str(terminal_id or "").strip()
    cid = _current_call_id.get()
    if not tid or not cid:
        return
    try:
        observer(cid, tid)
    except Exception:  # noqa: BLE001 - 进度通知失败不能升级成工具崩溃
        return


@dataclass(frozen=True)
class PlanItem:
    """计划清单里的一条（``todo_write`` 落盘之后的快照）。"""

    id: str
    content: str
    status: str


PlanObserver = Callable[[list[PlanItem]], None]
"""``(items) -> None``；``todo_write`` 落盘之后被调用，给的是整张表。"""

_plan_observer: ContextVar[PlanObserver | None] = ContextVar(
    "llgraph_plan_observer", default=None
)


def set_plan_observer(observer: PlanObserver | None) -> Token:
    """
    登记计划观察者。

    @param observer 观察者；None 表示不通知
    @return ContextVar token，交给 ``reset_plan_observer``
    """
    return _plan_observer.set(observer)


def reset_plan_observer(token: Token) -> None:
    """
    还原上一层观察者。

    @param token ``set_plan_observer`` 的返回值
    """
    _plan_observer.reset(token)


@contextmanager
def use_plan_observer(observer: PlanObserver | None) -> Iterator[None]:
    """
    在一段执行期间登记观察者（一轮 invoke 外面套一层）。

    @param observer 观察者；None 时等于什么都不做
    """
    token = set_plan_observer(observer)
    try:
        yield
    finally:
        reset_plan_observer(token)


def current_plan_observer() -> PlanObserver | None:
    """@return 当前观察者；无则 None"""
    return _plan_observer.get()


def notify_plan_updated(items: list[PlanItem]) -> None:
    """
    通知「任务清单现在是这样」。

    给的一律是**整张表**：入口那边（ACP 的 ``plan``）也是全量快照，
    报增量还得在两头各维护一份当前表，对不上就会画出一张与落盘不符的清单。
    清空清单同样要报（空列表），否则编辑器里那张表会停在被清空之前的样子。

    @param items 落盘后的全部条目，按清单顺序
    """
    observer = _plan_observer.get()
    if observer is None:
        return
    try:
        observer(list(items))
    except Exception:  # noqa: BLE001 - 进度通知失败不能升级成工具崩溃
        return
