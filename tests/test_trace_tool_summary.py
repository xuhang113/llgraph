"""Trace 工具调用摘要：grep 等须展示 pattern 而非仅 path。"""

from llgraph.display.trace_display import (
    _format_planned_tools_summary,
    _format_tool_step_title,
    _short_tool_target,
)


def test_grep_files_shows_pattern_not_only_dot_path() -> None:
    args = {"pattern": "export|Export|导出", "path": ".", "file_glob": "*.java"}
    assert _short_tool_target("grep_files", args) == "export|Export|导出, glob=*.java"


def test_grep_files_shows_path_when_not_workspace_root() -> None:
    args = {"pattern": "createExportTask", "path": "demo-api-service"}
    assert _short_tool_target("grep_files", args) == (
        "createExportTask, path=demo-api-service"
    )


def test_planned_tools_summary_distinguishes_parallel_greps() -> None:
    tool_calls = [
        {
            "name": "grep_files",
            "args": {"pattern": "export|Export", "path": ".", "file_glob": "*.java"},
        },
        {
            "name": "grep_files",
            "args": {"pattern": "userId.*orderId", "path": ".", "file_glob": "*.java"},
        },
    ]
    summary = _format_planned_tools_summary(tool_calls)
    assert "export|Export" in summary
    assert "userId.*orderId" in summary
    assert summary.count("grep_files(.)") == 0


def test_tool_step_title_includes_pattern() -> None:
    title = _format_tool_step_title(
        "grep_files",
        {"pattern": "class UserEntity", "path": ".", "file_glob": "**/UserEntity.java"},
    )
    assert title == "执行 grep_files(class UserEntity, glob=**/UserEntity.java)"
