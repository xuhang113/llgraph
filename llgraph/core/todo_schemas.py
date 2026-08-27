"""todo_write 入参。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TodoItemInput(BaseModel):
    """单条任务。"""

    id: str = Field(
        default="",
        description="稳定 id（如 t1）；空则按正文匹配已有项，否则自动分配",
    )
    content: str = Field(
        description="祈使句任务，如「给 search_replace 补回归测试」",
    )
    status: str = Field(
        default="pending",
        description="pending | in_progress | completed | cancelled；同时最多 1 条 in_progress",
    )


class TodoWriteInput(BaseModel):
    """todo_write 入参。"""

    todos: list[TodoItemInput] = Field(
        default_factory=list,
        description="要写入的任务；空且 merge=true 时返回当前清单。merge=true 按 id/正文 upsert",
    )
    merge: bool = Field(
        default=True,
        description="true=合并（推荐逐步更新）；false=用本次 todos 整表替换",
    )
