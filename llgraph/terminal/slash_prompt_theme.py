"""prompt_toolkit 斜杠补全样式：终端默认底色 + 分色 + 选中高亮。"""

from __future__ import annotations

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style

from llgraph.terminal.slash_catalog import SlashCatalogItem, slash_category_badge

# 覆盖 default_ui_style 的 completion-menu bg:#bbbbbb，使用终端默认底色（ansidefault）
SLASH_COMPLETION_STYLE = Style.from_dict(
    {
        "prompt": "bold ansigreen",
        "completion-menu": "bg:ansidefault fg:ansidefault noinherit",
        "completion-menu.completion": "bg:ansidefault noinherit",
        "completion-menu.completion.current": "bg:ansidefault reverse bold",
        "completion-menu.meta.completion": "bg:ansidefault noinherit",
        "completion-menu.meta.completion.current": "bg:ansidefault reverse bold",
        "slash-cmd": "bold ansicyan",
        "slash-badge-skill": "bold ansiyellow",
        "slash-badge-command": "bold ansimagenta",
        "slash-badge-meta": "bold ansibrightblack",
        "slash-desc": "#909090",
        "completion-menu.completion.current slash-cmd": "bold ansibrightcyan",
        "completion-menu.completion.current slash-badge-skill": "bold ansibrightyellow",
        "completion-menu.completion.current slash-badge-command": "bold ansibrightmagenta",
        "completion-menu.completion.current slash-badge-meta": "bold ansibrightblack",
        "completion-menu.completion.current slash-desc": "#cccccc",
        "scrollbar.background": "bg:ansidefault noinherit",
        "scrollbar.button": "bg:ansidefault reverse",
    }
)

_NAME_WIDTH = 28
_DESC_MAX = 72


def _badge_style_class(category: str) -> str:
    """
    类型标志对应样式 class。

    @param category Skills | Commands | 内置
    @return slash-badge-* class 名
    """
    if category == "Skills":
        return "slash-badge-skill"
    if category == "Commands":
        return "slash-badge-command"
    return "slash-badge-meta"


def format_slash_completion_display(item: SlashCatalogItem) -> FormattedText:
    """
    补全行：命令（青）+ 类型标志（黄/品红/灰）+ 说明（暗色）。

    @param item 目录项
    @return FormattedText（由 prompt_toolkit Style 着色，禁止内嵌 ANSI）
    """
    name = f"/{item.name}"
    badge = slash_category_badge(item.category)
    desc = item.description.strip()
    if len(desc) > _DESC_MAX:
        desc = desc[: _DESC_MAX - 1] + "…"
    pad = max(1, _NAME_WIDTH - len(name))
    return FormattedText(
        [
            ("class:slash-cmd", name),
            ("", " " * pad),
            (f"class:{_badge_style_class(item.category)}", badge),
            ("", "  "),
            ("class:slash-desc", desc),
        ]
    )


def format_slash_completion_meta(item: SlashCatalogItem) -> str:
    """
    副列留空（分色内容均在 display）。

    @param item 目录项
    @return 空串
    """
    return ""


def format_slash_completion_line(item: SlashCatalogItem) -> str:
    """
    纯文本斜杠补全一行。

    @param item 目录项
    @return 纯文本
    """
    name = f"/{item.name}"
    badge = slash_category_badge(item.category)
    desc = item.description.strip()
    if len(desc) > _DESC_MAX:
        desc = desc[: _DESC_MAX - 1] + "…"
    pad = max(1, _NAME_WIDTH - len(name))
    return f"{name}{' ' * pad}{badge}  {desc}"
