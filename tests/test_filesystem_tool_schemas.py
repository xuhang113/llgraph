"""内置文件工具 schema 与 pattern 别名测试。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.utils.function_calling import convert_to_openai_tool

from llgraph.core.filesystem_tool_schemas import GlobFilesInput, GrepFilesInput, ReadFileInput
from llgraph.core.filesystem_tools import create_filesystem_tools
from llgraph.core.workspace import WorkspaceContext


def test_glob_files_schema_has_field_descriptions() -> None:
    schema = convert_to_openai_tool(
        next(t for t in _tools() if t.name == "glob_files")
    )
    props = schema["function"]["parameters"]["properties"]
    assert "glob_pattern" in props
    assert "description" in props["glob_pattern"]
    assert "pattern" in props
    assert "description" in props["pattern"]


def test_grep_files_schema_uses_pattern_not_glob_pattern() -> None:
    schema = convert_to_openai_tool(
        next(t for t in _tools() if t.name == "grep_files")
    )
    props = schema["function"]["parameters"]["properties"]
    assert "pattern" in props
    assert "description" in props["pattern"]
    assert "glob_pattern" not in props


def test_glob_files_input_accepts_pattern_alias() -> None:
    parsed = GlobFilesInput.model_validate({"pattern": "**/UserEntity.java"})
    assert parsed.glob_pattern == "**/UserEntity.java"
    assert parsed.path == "."


def test_glob_files_tool_invoke_with_pattern_alias(tmp_path: Path) -> None:
    (tmp_path / "UserEntity.java").write_text("class UserEntity {}", encoding="utf-8")
    tool = next(t for t in _tools(tmp_path) if t.name == "glob_files")
    out = tool.invoke({"pattern": "**/UserEntity.java"})
    assert "UserEntity.java" in out
    assert "失败" not in out


def test_glob_files_tool_invoke_with_glob_pattern(tmp_path: Path) -> None:
    (tmp_path / "Foo.java").write_text("class Foo {}", encoding="utf-8")
    tool = next(t for t in _tools(tmp_path) if t.name == "glob_files")
    out = tool.invoke({"glob_pattern": "**/Foo.java"})
    assert "Foo.java" in out


def test_grep_files_input_rejects_empty_pattern() -> None:
    parsed = GrepFilesInput.model_validate({"pattern": "UserEntity"})
    assert parsed.pattern == "UserEntity"
    assert parsed.output_mode == "auto"
    assert parsed.head_limit == 0


def test_grep_files_schema_includes_output_mode() -> None:
    schema = convert_to_openai_tool(
        next(t for t in _tools() if t.name == "grep_files")
    )
    props = schema["function"]["parameters"]["properties"]
    assert "output_mode" in props
    assert "head_limit" in props
    assert "auto" in props["output_mode"]["description"]


def test_search_replace_schema_includes_replacements() -> None:
    ctx = WorkspaceContext(Path(".").resolve(), allow_write=True)
    tools = create_filesystem_tools(ctx)
    schema = convert_to_openai_tool(next(t for t in tools if t.name == "search_replace"))
    props = schema["function"]["parameters"]["properties"]
    assert "old_string" in props
    assert "replacements" in props
    assert "replace_all" in props


def test_read_file_schema_mentions_outline_fold() -> None:
    schema = convert_to_openai_tool(
        next(t for t in _tools() if t.name == "read_file")
    )
    props = schema["function"]["parameters"]["properties"]
    assert "大纲" in props["end_line"]["description"] or "折叠" in props["end_line"]["description"]
    parsed = ReadFileInput.model_validate({"path": "a.py"})
    assert parsed.start_line == 1
    assert parsed.end_line == 0


def _tools(root: Path | None = None):
    ws = root or Path(".").resolve()
    ctx = WorkspaceContext(ws, allow_write=False)
    return create_filesystem_tools(ctx)
