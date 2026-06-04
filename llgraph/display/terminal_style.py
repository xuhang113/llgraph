"""经典终端 Banner / 菜单：分组、树形缩进、ANSI 配色。"""

from __future__ import annotations

from pathlib import Path

from llgraph.ui.style import indent_line, sty

LABEL_COL_WIDTH = 14
_TREE_BRANCH = "├─"
_TREE_LAST = "└─"
_SECTION_MARK = "▸"


def print_section(title: str) -> None:
    """
    打印分组标题（▸ 工作区）。

    @param title 分组名
    """
    print(sty(f"{_SECTION_MARK} {title}", "brand"), flush=True)


def _style_banner_value(label: str, value: str) -> str:
    """
    按字段语义着色 value。

    @param label 字段名
    @param value 展示值
    @return 着色后的 value
    """
    if label in ("workspace", "会话记忆"):
        return sty(value, "path")
    if label in ("thread", "文件"):
        return sty(value, "accent")
    if label in ("Code index",):
        return sty(value, "ok")
    if label in ("Index watch", "MCP") and ("未" in value or "失败" in value):
        return sty(value, "warn")
    if label in ("Index watch",) and "已启动" in value:
        return sty(value, "ok")
    if label in ("LLM 模型", "向量检索日志"):
        return sty(value, "value")
    return sty(value, "value")


def print_tree_row(
    label: str,
    value: str,
    *,
    hint: str = "",
    is_last: bool = False,
) -> None:
    """
    打印树形 key-value 行。

    @param label 字段名
    @param value 值
    @param hint 灰色括号说明
    @param is_last 是否最后一行（└─）
    """
    branch = _TREE_LAST if is_last else _TREE_BRANCH
    label_part = sty(f"{label:<{LABEL_COL_WIDTH}}", "label")
    line = (
        indent_line(1)
        + sty(branch, "hint")
        + " "
        + label_part
        + " "
        + _style_banner_value(label, value)
    )
    if hint:
        line += " " + sty(f"({hint})", "hint")
    print(line, flush=True)


def print_section_rows(
    rows: list[tuple[str, str, str]],
) -> None:
    """
    打印一组树形行。

    @param rows (label, value, hint) 列表
    """
    if not rows:
        return
    for idx, (label, value, hint) in enumerate(rows):
        print_tree_row(label, value, hint=hint, is_last=idx == len(rows) - 1)
