"""斜杠补全专用：全宽菜单（不影响 PromptSession 默认 CompletionsMenu）。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from prompt_toolkit.filters import FilterOrBool, has_completions, is_done, to_filter
from prompt_toolkit.layout.containers import ConditionalContainer, Float, FloatContainer, Window
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu, CompletionsMenuControl
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.layout.scrollable_pane import ScrollOffsets

if TYPE_CHECKING:
    from prompt_toolkit.layout.containers import AnyContainer
    from prompt_toolkit.shortcuts import PromptSession


class SlashFullWidthCompletionsMenu(ConditionalContainer):
    """全终端宽度的补全列表（透明 Float，不铺 prompt_toolkit 默认灰底）。"""

    def __init__(
        self,
        *,
        max_height: int | None = 16,
        scroll_offset: int | Callable[[], int] = 0,
        menu_filter: FilterOrBool | None = None,
        z_index: int = 10**8,
    ) -> None:
        show_filter = (
            to_filter(menu_filter)
            if menu_filter is not None
            else (has_completions & ~is_done)
        )
        super().__init__(
            content=Window(
                content=CompletionsMenuControl(),
                height=Dimension(min=1, max=max_height),
                scroll_offsets=ScrollOffsets(top=scroll_offset, bottom=scroll_offset),
                right_margins=[ScrollbarMargin()],
                dont_extend_width=False,
                z_index=z_index,
            ),
            filter=show_filter,
        )


def _walk_containers(container: AnyContainer, visit: Callable[[AnyContainer], None]) -> None:
    """
    深度遍历 layout 容器树。

    @param container 根容器
    @param visit 访问回调
    """
    seen: set[int] = set()

    def _walk(node: AnyContainer) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        visit(node)
        if isinstance(node, FloatContainer):
            _walk(node.content)
            for fl in node.floats:
                _walk(fl.content)
            return
        if isinstance(node, ConditionalContainer):
            _walk(node.content)
            if node.alternative_content is not None:
                _walk(node.alternative_content)
            return
        # Frame 等带 body 的容器
        body = getattr(node, "body", None)
        if body is not None:
            _walk(body)
            return
        children = getattr(node, "children", None)
        if children:
            for child in children:
                _walk(child)
            return
        inner = getattr(node, "content", None)
        if inner is not None and inner is not node:
            _walk(inner)

    _walk(container)


def patch_slash_completion_menu_full_width(session: PromptSession) -> None:
    """
    将会话内 COLUMN 补全菜单替换为全宽 Float（仅斜杠 PromptSession 调用）。

    @param session prompt_toolkit PromptSession
    """

    def _replace_floats(container: AnyContainer) -> None:
        if not isinstance(container, FloatContainer):
            return
        replaced: list[Float] = []
        for fl in container.floats:
            if isinstance(fl.content, CompletionsMenu):
                replaced.append(
                    Float(
                        left=0,
                        right=0,
                        ycursor=True,
                        transparent=True,
                        content=SlashFullWidthCompletionsMenu(
                            max_height=16,
                            scroll_offset=1,
                            menu_filter=fl.content.filter,
                        ),
                    )
                )
            else:
                replaced.append(fl)
        container.floats[:] = replaced

    _walk_containers(session.layout.container, _replace_floats)
