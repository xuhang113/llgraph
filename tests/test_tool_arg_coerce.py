"""工具入参纠偏：Claude Code / Cursor 字段名与宽松类型。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt.tool_node import ToolNode

from llgraph.core.filesystem_tool_schemas import (
    GrepFilesInput,
    ReadFileInput,
    ReadFilesInput,
    SearchReplaceInput,
    WriteFileInput,
)
from llgraph.core.filesystem_tools import create_filesystem_tools
from llgraph.core.shell_schemas import RunShellCommandInput
from llgraph.core.todo_schemas import TodoWriteInput
from llgraph.core.tool_arg_coerce import (
    coerce_tool_args,
    coerce_tool_call,
    wrap_tool_node_with_arg_coerce,
)
from llgraph.core.tool_loop_guard import compute_blocked_tool_messages
from llgraph.core.workspace import WorkspaceContext
from llgraph.core.write_serialize import write_path_from_call


def _tools(root: Path, *, allow_write: bool = False):
    ctx = WorkspaceContext(root, allow_write=allow_write)
    return {tool.name: tool for tool in create_filesystem_tools(ctx)}


def test_coerce_read_file_claude_code_aliases() -> None:
    out = coerce_tool_args(
        "read_file",
        {"file_path": "src/Foo.java", "offset": 20, "limit": 10},
    )
    assert out["path"] == "src/Foo.java"
    assert out["start_line"] == 20
    assert out["end_line"] == 29
    assert "file_path" not in out
    assert "offset" not in out


def test_coerce_read_file_schema_and_tool(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    parsed = ReadFileInput.model_validate(
        {"file_path": "a.py", "offset": 2, "limit": 2}
    )
    assert parsed.path == "a.py"
    assert parsed.start_line == 2
    assert parsed.end_line == 3
    tools = _tools(tmp_path)
    out = tools["read_file"].invoke({"file_path": "a.py", "offset": 2, "limit": 2})
    assert "a.py" in out
    assert "b" in out
    assert "错误" not in out


def test_coerce_read_files_string_and_json_paths() -> None:
    listed = ReadFilesInput.model_validate({"paths": "src/a.py, src/b.py"})
    assert listed.paths == ["src/a.py", "src/b.py"]
    json_paths = ReadFilesInput.model_validate({"paths": '["pkg/a.py", "pkg/b.py"]'})
    assert json_paths.paths == ["pkg/a.py", "pkg/b.py"]
    single = ReadFilesInput.model_validate({"path": "only.py"})
    assert single.paths == ["only.py"]


def test_coerce_read_files_tool_invoke(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("B = 2\n", encoding="utf-8")
    tools = _tools(tmp_path)
    out = tools["read_files"].invoke({"paths": "a.py\nb.py"})
    assert "A = 1" in out
    assert "B = 2" in out


def test_coerce_grep_glob_alias() -> None:
    parsed = GrepFilesInput.model_validate(
        {"query": "FooService", "glob": "*.java", "path": "."}
    )
    assert parsed.pattern == "FooService"
    assert parsed.file_glob == "*.java"


def test_coerce_search_replace_cursor_and_claude_fields() -> None:
    parsed = SearchReplaceInput.model_validate(
        {
            "file_path": "src/a.py",
            "oldString": "foo",
            "newString": "bar",
            "replaceAll": True,
        }
    )
    assert parsed.path == "src/a.py"
    assert parsed.old_string == "foo"
    assert parsed.new_string == "bar"
    assert parsed.replace_all is True


def test_coerce_search_replace_replacements_json_string() -> None:
    parsed = SearchReplaceInput.model_validate(
        {
            "path": "a.py",
            "replacements": '[{"old_string":"a","new_string":"b"}]',
        }
    )
    assert len(parsed.replacements) == 1
    assert parsed.replacements[0].old_string == "a"


def test_coerce_write_file_schema() -> None:
    parsed = WriteFileInput.model_validate(
        {"file_path": "docs/n.md", "contents": "# hi"}
    )
    assert parsed.path == "docs/n.md"
    assert parsed.content == "# hi"


def test_coerce_write_file_tool_invoke(tmp_path: Path) -> None:
    tools = _tools(tmp_path, allow_write=True)
    out = tools["write_file"].invoke({"file_path": "n.md", "contents": "hello"})
    assert "已写入" in out
    assert (tmp_path / "n.md").read_text(encoding="utf-8") == "hello"


def test_coerce_shell_command_list_and_timeout() -> None:
    parsed = RunShellCommandInput.model_validate(
        {"command": ["ls", "-la"], "timeout": 30, "cwd": "src"}
    )
    assert parsed.command == "ls -la"
    assert parsed.block_until_ms == 30_000
    assert parsed.working_directory == "src"
    bg = RunShellCommandInput.model_validate(
        {"cmd": "sleep 1", "run_in_background": True}
    )
    assert bg.command == "sleep 1"
    assert bg.block_until_ms == 0


def test_coerce_todo_json_string() -> None:
    parsed = TodoWriteInput.model_validate(
        {"todos": '[{"content":"改 search_replace","status":"in_progress"}]'}
    )
    assert len(parsed.todos) == 1
    assert parsed.todos[0].content == "改 search_replace"


def test_coerce_nested_arguments_wrapper() -> None:
    out = coerce_tool_args(
        "read_file",
        {"arguments": {"file_path": "a.py", "offset": "4"}},
    )
    assert out["path"] == "a.py"
    assert out["start_line"] == 4


def test_loop_guard_treats_file_path_as_same_read() -> None:
    first = {
        "id": "c1",
        "name": "read_file",
        "args": {"file_path": "src/a.py"},
        "type": "tool_call",
    }
    msgs = [
        HumanMessage(content="读"),
        AIMessage(content="", tool_calls=[first]),
        ToolMessage(content="--- src/a.py (行 1-3 / 共 3 行) ---\n1| x", tool_call_id="c1", name="read_file"),
        AIMessage(
            content="",
            tool_calls=[{"id": "c2", "name": "read_file", "args": {"path": "src/a.py"}}],
        ),
    ]
    blocked = compute_blocked_tool_messages(
        msgs, [{"id": "c2", "name": "read_file", "args": {"path": "src/a.py"}}]
    )
    assert "c2" in blocked


def test_write_serialize_groups_file_path_with_path() -> None:
    from llgraph.core.write_serialize import gate_from_tool_calls

    gate = gate_from_tool_calls(
        [
            {
                "id": "c1",
                "name": "search_replace",
                "args": {"file_path": "a.py", "old_string": "x"},
            },
            {
                "id": "c2",
                "name": "search_replace",
                "args": {"path": "a.py", "old_string": "y"},
            },
        ]
    )
    assert gate._id_to_path["c1"] == "a.py"
    assert gate._id_to_path["c2"] == "a.py"


def test_tool_node_executes_file_path_alias(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    (tmp_path / "hit.py").write_text("HIT = 1\n", encoding="utf-8")
    tool = _tools(tmp_path)["read_file"]
    inner = ToolNode([tool])
    wrap_tool_node_with_arg_coerce(inner)
    runtime = MagicMock()
    runtime.config = {}
    runtime.state = {}
    call = {
        "id": "r1",
        "name": "read_file",
        "args": {"file_path": "hit.py"},
        "type": "tool_call",
    }
    msg = inner._run_one(call, "list", runtime)
    body = str(getattr(msg, "content", msg))
    assert "HIT = 1" in body
    assert "错误" not in body


def test_validation_error_returns_tool_message(tmp_path: Path) -> None:
    tools = _tools(tmp_path, allow_write=True)
    out = tools["search_replace"].invoke({"path": "a.py"})
    assert isinstance(out, str)
    assert "错误" in out
    assert "old_string" in out or "replacements" in out or "参数无效" in out


def test_coerce_tool_call_copies_args() -> None:
    original = {
        "id": "c1",
        "name": "read_file",
        "args": {"file_path": "a.py"},
        "type": "tool_call",
    }
    patched = coerce_tool_call(original)
    assert patched["args"]["path"] == "a.py"
    assert original["args"]["file_path"] == "a.py"
    assert "path" not in original["args"]
