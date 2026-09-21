"""编辑器里那份文件：未保存的缓冲区。

磁盘上是「上次保存的样子」，用户正在编辑器里改的那份可能还没落盘。
ACP 客户端声明 ``fs/readTextFile`` / ``fs/writeTextFile`` 时，文件工具就多了一条来源：
读以缓冲区为准，改一个**有未保存改动**的文件也交回编辑器写——
我们自己原子替换的话，用户随后在编辑器里按一次保存就把这次修改盖回去了。

来源放在 ContextVar 上，理由与 `permissions/approval.py` 的闸门一样：
工具可能在 LangGraph 的线程池里跑，全局变量会在多会话同进程时串台。

没有入口登记来源时一律回落磁盘——CLI / Web Console 的行为因此完全不变。

路径判定（沙箱、越界、只读模式、文件类型）仍然全部走磁盘那条：
编辑器只回内容，不回「这个路径能不能动」。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class EditorFileSource(Protocol):
    """编辑器侧的文本文件读写（ACP 下是两条反向请求）。"""

    def read_text(self, path: Path) -> str | None:
        """@param path 绝对路径 @return 编辑器里的正文；拿不到时 None"""

    def write_text(self, path: Path, text: str) -> bool:
        """@param path 绝对路径 @param text 全量正文 @return 是否已由编辑器写下"""


_editor_files: ContextVar[EditorFileSource | None] = ContextVar(
    "llgraph_editor_files", default=None
)


def set_editor_file_source(source: EditorFileSource | None) -> Token:
    """
    登记编辑器文件来源。

    @param source 来源；None 表示只走磁盘
    @return ContextVar token，交给 ``reset_editor_file_source``
    """
    return _editor_files.set(source)


def reset_editor_file_source(token: Token) -> None:
    """
    还原上一层来源。

    @param token ``set_editor_file_source`` 的返回值
    """
    _editor_files.reset(token)


@contextmanager
def use_editor_file_source(source: EditorFileSource | None) -> Iterator[None]:
    """
    在一段执行期间登记来源（一轮 invoke 外面套一层）。

    @param source 来源；None 时等于什么都不做
    """
    token = set_editor_file_source(source)
    try:
        yield
    finally:
        reset_editor_file_source(token)


def current_editor_file_source() -> EditorFileSource | None:
    """@return 当前来源；无则 None"""
    return _editor_files.get()


def editor_buffer_text(path: Path) -> str | None:
    """
    取编辑器里的正文。

    读不到一律当「没有这条来源」：缓冲区只是更准，不是必需，
    编辑器答不上来时回落磁盘比让工具报错强。

    @param path 绝对路径
    @return 正文；没有来源或读不到时 None
    """
    source = _editor_files.get()
    if source is None:
        return None
    try:
        text = source.read_text(Path(path))
    except Exception:  # noqa: BLE001 - 编辑器侧出问题不能升级成工具崩溃
        return None
    return text if isinstance(text, str) else None


def editor_write_text(path: Path, text: str) -> bool:
    """
    让编辑器写这份正文。

    @param path 绝对路径
    @param text 全量正文
    @return 是否已由编辑器写下；False 时调用方必须自己落盘
    """
    source = _editor_files.get()
    if source is None:
        return False
    try:
        return bool(source.write_text(Path(path), text))
    except Exception:  # noqa: BLE001 - 写不下就退回磁盘，不能丢掉这次修改
        return False
