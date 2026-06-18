"""Plan 模式终端 Banner。"""

from __future__ import annotations

from pathlib import Path

from llgraph.display.trace_display import TRACE_MODE_LABELS, TraceSession


def print_plan_session_banner(
    *,
    workspace: Path,
    thread_id: str,
    trace_session: TraceSession,
    allow_write: bool,
    phase: str = "",
    task_progress: str = "",
) -> None:
    """
    打印 Plan 模式启动 Banner。

    @param workspace 工作区
    @param thread_id plan thread
    @param trace_session 追踪配置
    @param allow_write CLI -w
    @param phase 当前 phase
    @param task_progress task 进度摘要
    """
    from llgraph.display.terminal_style import print_section, print_section_rows
    from llgraph.terminal.style import sty

    ws = str(workspace.expanduser().resolve())
    write_hint = "CLI -w 已开（仍须在 Confirm Survey 勾选写权限）" if allow_write else "默认只读"
    status_line = phase or "starting"
    if task_progress:
        status_line = f"{status_line} · {task_progress}"

    print(sty("llgraph plan — 多 Agent 编排", "brand"), flush=True)
    print(sty("  /plan · /session agent · /trace · /help · exit", "hint"), flush=True)
    print("", flush=True)
    print_section("Plan 会话")
    print_section_rows(
        [
            ("workspace", ws, ""),
            ("thread", thread_id, ""),
            ("状态", status_line, ""),
            ("写权限", write_hint, "Confirm Survey 控制 Worker 写"),
            ("过程展示", TRACE_MODE_LABELS[trace_session.mode], "/trace"),
        ]
    )
    print("", flush=True)
