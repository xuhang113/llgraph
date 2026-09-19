"""写入 / 执行前的逐次授权：把「这一刀要不要放行」交给当前入口决定。

`file_write.py`、`shell.py` 是**策略**判定（这个模式允不允许这类操作），
这里是**交互**：策略放行之后，再问一次当前坐在前面的人。

没有入口登记授权闸门时一律放行——CLI / Web Console 的行为因此完全不变；
ACP（编辑器）在每轮 invoke 外面登记一个闸门，于是写工具会变成编辑器里的授权弹窗。

闸门放在 ContextVar 上而不是全局变量：工具可能在 LangGraph 的线程池里跑，
langchain 提交任务时会复制调用方的 context，全局变量则会在多会话同进程时串台。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Callable, Iterator

APPROVAL_KIND_EDIT = "edit"
"""改工作区文件。"""

APPROVAL_KIND_EXECUTE = "execute"
"""跑 shell 命令。"""

_MAX_TITLE_CHARS = 160


@dataclass
class ApprovalRequest:
    """一次待授权的工具动作。"""

    tool: str
    kind: str = APPROVAL_KIND_EDIT
    path: str | None = None
    old_text: str | None = None
    new_text: str | None = None
    command: str | None = None
    cwd: str | None = None

    def target(self) -> str:
        """@return 动作对象（路径或命令），用于提示文案"""
        if self.path:
            return self.path
        one_line = " ".join((self.command or "").split())
        if len(one_line) > _MAX_TITLE_CHARS:
            return one_line[:_MAX_TITLE_CHARS] + "…"
        return one_line

    def title(self) -> str:
        """@return 一行标题（编辑器弹窗标题，沿用 trace 的「执行 工具(对象)」样式）"""
        target = self.target()
        return f"执行 {self.tool}({target})" if target else f"执行 {self.tool}"


@dataclass
class ApprovalDecision:
    """授权结果。"""

    allowed: bool
    cancelled: bool = False
    reason: str = ""


_ALLOWED = ApprovalDecision(allowed=True)

ApprovalAsk = Callable[[ApprovalRequest], ApprovalDecision]
"""闸门：收到一次动作，回一个决定（会阻塞，比如等编辑器里的人点按钮）。"""

_approval_gate: ContextVar[ApprovalAsk | None] = ContextVar(
    "llgraph_approval_gate", default=None
)


def set_approval_gate(ask: ApprovalAsk | None) -> Token:
    """
    登记授权闸门。

    @param ask 闸门；None 表示不需要授权（直接放行）
    @return ContextVar token，交给 ``reset_approval_gate``
    """
    return _approval_gate.set(ask)


def reset_approval_gate(token: Token) -> None:
    """
    还原上一层闸门。

    @param token ``set_approval_gate`` 的返回值
    """
    _approval_gate.reset(token)


@contextmanager
def use_approval_gate(ask: ApprovalAsk | None) -> Iterator[None]:
    """
    在一段执行期间登记闸门（一轮 invoke 外面套一层）。

    @param ask 闸门；None 时等于什么都不做
    """
    token = set_approval_gate(ask)
    try:
        yield
    finally:
        reset_approval_gate(token)


def current_approval_gate() -> ApprovalAsk | None:
    """@return 当前闸门；无则 None"""
    return _approval_gate.get()


def request_approval(req: ApprovalRequest) -> ApprovalDecision:
    """
    问一次闸门要不要放行。

    闸门自己抛异常（编辑器断了、回包读不懂）按拒绝算：授权链路出问题时
    宁可这一刀不落地，也不能默认放行。

    @param req 动作
    @return 决定
    """
    ask = _approval_gate.get()
    if ask is None:
        return _ALLOWED
    try:
        decision = ask(req)
    except Exception as exc:  # noqa: BLE001 - 授权失败不能升级成工具崩溃
        return ApprovalDecision(allowed=False, reason=f"授权请求失败: {exc}")
    if not isinstance(decision, ApprovalDecision):
        return ApprovalDecision(allowed=False, reason="授权回复无法识别")
    return decision


def check_approval(req: ApprovalRequest) -> str | None:
    """
    授权检查，给工具直接用。

    @param req 动作
    @return None 表示放行；否则是要回给模型的说明
    """
    decision = request_approval(req)
    if decision.allowed:
        return None
    target = req.target()
    where = f"（{target}）" if target else ""
    if decision.cancelled:
        return f"已停止本轮：{req.tool}{where} 未执行。"
    tail = f"原因：{decision.reason}" if decision.reason else ""
    return (
        f"用户拒绝了这次 {req.tool}{where}，未做任何改动。{tail}"
        "不要重复同一次调用，先说明你打算怎么改并等用户答复。"
    )
