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
    end_line: int = Field(default=0, description="结束行号（含）；0 表示读到文件末尾")


class ReadFilesInput(BaseModel):
    """read_files 入参。"""

    paths: list[str] = Field(
        description="多个完整相对路径的数组，如 [\"src/a.java\", \"src/b.java\"]，最多 8 个",
    )
    start_line: int = Field(default=1, description="每个文件的起始行号，从 1 开始")
    end_line: int = Field(default=0, description="每个文件的结束行号（含）；0 表示到末尾")
