"""内置 shell 工具 Pydantic 入参。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RunShellCommandInput(BaseModel):
    """run_shell_command 入参。"""

    command: str = Field(description="要执行的 shell 命令（单条）")
    working_directory: str = Field(
        default="",
        description=(
            "相对工作区的执行目录；空表示沿用本会话上次目录（初始为工作区根）。"
            "显式传 . 则回到工作区根。"
        ),
    )
    block_until_ms: int | None = Field(
        default=None,
        description=(
            "前台等待毫秒。省略则等到 timeout_sec 后终止并返回已捕获输出；"
            "0=立即后台并返回 job_id；>0 则等待该时长，仍在运行就转后台。"
            "对齐 Cursor run_terminal_cmd / Claude Code Bash。"
        ),
    )


class AwaitShellInput(BaseModel):
    """await_shell 入参。"""

    job_id: str = Field(description='后台任务 id，如 sh-a1b2c3（run_shell_command 返回）')
    block_until_ms: int = Field(
        default=30_000,
        description="本次最多再等多少毫秒；0 表示立即返回当前输出",
    )
    pattern: str = Field(
        default="",
        description="可选正则；在已捕获输出中匹配到则提前返回（服务就绪探测）",
    )
