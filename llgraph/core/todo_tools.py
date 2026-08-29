"""todo_write 工具：会话级任务清单（对标 Cursor / Claude Code TodoWrite）。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import StructuredTool

from llgraph.context.runtime_context import get_active_thread_id
from llgraph.core.todo_schemas import TodoWriteInput
from llgraph.core.todo_store import (
    TODO_TOOL_NAME,
    apply_todo_write,
    empty_todo_state,
    format_todo_tool_result,
    load_todo_state,
    parse_todo_inputs,
    save_todo_state,
)
from llgraph.core.tool_arg_coerce import format_tool_validation_error


def create_todo_tools(workspace_root: Path) -> list:
    """
    创建 todo_write。

    @param workspace_root 工作区根
    @return Tool 列表
    """
    root = workspace_root.expanduser().resolve()

    def todo_write(
        todos: list | None = None,
        merge: bool = True,
    ) -> str:
        """
        写入本会话任务清单（对标 Cursor 任务列表 / Claude Code TodoWrite）。

        两步以上的改码/排查：动手前先列出步骤，始终最多 1 条 in_progress。
        清单落在会话侧，压缩与 tool 裁剪不会丢掉。完成或取消一项立刻 merge 更新。
        单步问答不要建清单。

        @param todos 任务列表（id/content/status）
        @param merge true 按 id 合并；false 整表替换
        """
        thread_id = get_active_thread_id()
        if not thread_id:
            return "错误: 当前无活动会话，无法写入任务清单。"
        incoming = parse_todo_inputs(todos or [])
        if merge and not incoming:
            current = load_todo_state(root, thread_id)
            return format_todo_tool_result(current, ["未提供 todos，已返回当前清单。"])
        current = load_todo_state(root, thread_id) if merge else empty_todo_state()
        state, notes = apply_todo_write(current, incoming, merge=bool(merge))
        save_todo_state(root, thread_id, state)
        return format_todo_tool_result(state, notes)

    return [
        StructuredTool.from_function(
            func=todo_write,
            name=TODO_TOOL_NAME,
            description=todo_write.__doc__ or "",
            args_schema=TodoWriteInput,
            handle_validation_error=format_tool_validation_error,
        )
    ]
