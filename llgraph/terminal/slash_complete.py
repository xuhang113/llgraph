"""斜杠补全：prompt_toolkit 过滤逻辑。"""

from __future__ import annotations

from pathlib import Path

from llgraph.terminal.slash_catalog import (
    SlashCatalogItem,
    build_slash_catalog,
    filter_slash_catalog,
    parse_slash_partial,
)

try:
    from prompt_toolkit.completion import CompleteEvent, Completer, Completion
    from prompt_toolkit.document import Document

    from llgraph.terminal.slash_prompt_theme import (
        format_slash_completion_display,
        format_slash_completion_meta,
    )

    _HAS_PROMPT_TOOLKIT = True
except ImportError:
    _HAS_PROMPT_TOOLKIT = False


def prompt_toolkit_available() -> bool:
    """是否可用 prompt_toolkit 做终端补全。"""
    return _HAS_PROMPT_TOOLKIT


class SlashCompleter(Completer):
    """输入以 / 开头且尚无空格时，动态匹配 Skills / Commands / 内置。"""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.expanduser().resolve()
        self._catalog: list[SlashCatalogItem] | None = None

    def _items(self) -> list[SlashCatalogItem]:
        if self._catalog is None:
            self._catalog = build_slash_catalog(self._workspace)
        return self._catalog

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        partial = parse_slash_partial(document.text_before_cursor)
        if partial is None:
            return
        for item in filter_slash_catalog(self._items(), partial, limit=16):
            yield Completion(
                f"{item.name} ",
                start_position=-len(partial),
                display=format_slash_completion_display(item),
                display_meta=format_slash_completion_meta(item),
            )


def build_slash_completer(workspace: Path) -> SlashCompleter:
    """
    构建 prompt_toolkit Completer。

    @param workspace 工作区根
    @return SlashCompleter
    """
    return SlashCompleter(workspace)

