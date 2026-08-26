"""内置文件工具 Pydantic 入参（字段 description + glob pattern 别名兼容）。"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class GlobFilesInput(BaseModel):
    """glob_files 入参。"""

    glob_pattern: str = Field(
        default="",
        description='文件名/路径 glob，如 **/UserEntity.java、**/*.sh（首选此参数名）',
    )
    pattern: str = Field(
        default="",
        description="兼容误传：等同 glob_pattern（grep_files 才用 pattern 搜内容）；优先 glob_pattern",
    )
    path: str = Field(
        default=".",
        description="搜索根目录，相对工作区；默认 . 表示全工作区",
    )

    @model_validator(mode="after")
    def resolve_glob_pattern(self) -> GlobFilesInput:
        effective = self.glob_pattern.strip() or self.pattern.strip()
        if not effective:
            msg = "glob_pattern 必填（勿将 grep_files 的 pattern 当作 glob 参数名而不传 glob_pattern）"
            raise ValueError(msg)
        return self.model_copy(update={"glob_pattern": effective})


class GrepFilesInput(BaseModel):
    """grep_files 入参。"""

    pattern: str = Field(
        description="内容搜索模式（正则或字面量）；glob_files 找文件名用 glob_pattern",
    )
    path: str = Field(
        default=".",
        description="搜索根目录，相对工作区；默认 .",
    )
    file_glob: str = Field(
        default="",
        description='可选文件名限制，如 *.java、*.md；空表示不限制扩展名',
    )
    output_mode: str = Field(
        default="auto",
        description=(
            "结果形态：auto（默认，命中过多时折叠为文件统计+样例，对齐 Cursor/Claude Code）；"
            "content 逐行命中；files / files_with_matches 仅路径+命中数；"
            "count / count_matches 仅统计。"
        ),
    )
    head_limit: int = Field(
        default=0,
        description="content 模式最多返回的命中条数；0 表示默认 40",
    )


class ListDirectoryInput(BaseModel):
    """list_directory 入参。"""

    path: str = Field(
        default=".",
        description='相对工作区的目录路径，如 docs、.llgraph/context/tool-results',
    )


class ReadFileInput(BaseModel):
    """read_file 入参。"""

    path: str = Field(description="单个文件路径，相对工作区或 ~/.llgraph/skills|rules")
    start_line: int = Field(default=1, description="起始行号，从 1 开始")
    end_line: int = Field(
        default=0,
        description=(
            "结束行号（含）；0 表示到末尾。"
            "大文件且未指定行段时自动返回符号大纲+检索命中窗，而非全文；"
            "精读请给出完整函数/类的 start_line/end_line"
        ),
    )


class ReadFilesInput(BaseModel):
    """read_files 入参。"""

    paths: list[str] = Field(
        description="多个完整相对路径的数组，如 [\"src/a.java\", \"src/b.java\"]，最多 8 个",
    )
    start_line: int = Field(default=1, description="每个文件的起始行号，从 1 开始")
    end_line: int = Field(
        default=0,
        description=(
            "每个文件的结束行号（含）；0 表示到末尾。"
            "大文件未指定行段时同样折叠为大纲+命中窗"
        ),
    )


class SearchReplaceHunkInput(BaseModel):
    """search_replace.replacements 中的单个 hunk。"""

    old_string: str = Field(description="待替换片段")
    new_string: str = Field(default="", description="替换后文本；空字符串表示删除该片段")
    replace_all: bool = Field(default=False, description="是否替换该 hunk 的全部命中")


class SearchReplaceInput(BaseModel):
    """search_replace 入参。"""

    path: str = Field(description="相对工作区的文件路径")
    old_string: str = Field(
        default="",
        description="待替换片段；可与 replacements 同时提供（作为第一 hunk）",
    )
    new_string: str = Field(default="", description="替换后文本")
    replace_all: bool = Field(
        default=False,
        description="是否替换 old_string 的全部命中；多处且未设 true 时若要求唯一则失败",
    )
    replacements: list[SearchReplaceHunkInput] = Field(
        default_factory=list,
        description=(
            "同一文件多个 hunk，按顺序应用（对齐 Codex apply_patch / Cursor 多处编辑）。"
            "可避免对同一 path 并行多次 search_replace。"
        ),
    )

    @model_validator(mode="after")
    def require_hunk(self) -> SearchReplaceInput:
        if not (self.old_string or "").strip() and not self.replacements:
            raise ValueError("必须提供 old_string 或 replacements")
        return self
