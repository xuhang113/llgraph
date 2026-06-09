"""文件写权限：-w / allow_write 模式校验。"""

from __future__ import annotations

FILE_WRITE_DENIED_MESSAGE = (
    "当前为只读模式，不能写入文件。请使用 llgraph -w 启动以允许写入。"
)


def require_file_write(*, allow_write: bool) -> None:
    """
    写工具调用前校验是否已启用可写模式。

    @param allow_write 是否 -w / /write on
    """
    if not allow_write:
        raise PermissionError(FILE_WRITE_DENIED_MESSAGE)
