"""写入后语法诊断回注。"""

from __future__ import annotations

from pathlib import Path

from llgraph.config.edit_settings import resolve_edit_settings
from llgraph.core.edit_diagnostics import (
    DIAGNOSTIC_MARKER,
    collect_syntax_issues,
    format_edit_diagnostics,
)
from llgraph.core.filesystem_tools import create_filesystem_tools
from llgraph.core.workspace import WorkspaceContext


def test_python_syntax_error_is_collected() -> None:
    issues = collect_syntax_issues("a.py", "def foo(\n")
    assert issues
    assert issues[0].line >= 1
    assert "SyntaxError" in issues[0].message or "expected" in issues[0].message.lower() or "(" in issues[0].message


def test_valid_python_has_no_issues() -> None:
    assert collect_syntax_issues("a.py", "def foo():\n    return 1\n") == []


def test_json_syntax_error_is_collected() -> None:
    issues = collect_syntax_issues("a.json", '{"a": 1,}')
    assert issues
    assert issues[0].line >= 1


def test_format_skips_preexisting_errors() -> None:
    broken = "def foo(\n"
    still = format_edit_diagnostics("a.py", broken, old_text=broken)
    assert still == ""


def test_format_reports_new_python_error() -> None:
    old = "def foo():\n    return 1\n"
    new = "def foo(\n    return 1\n"
    block = format_edit_diagnostics("mod.py", new, old_text=old)
    assert DIAGNOSTIC_MARKER in block
    assert "mod.py:L" in block
    assert "声称已完成" in block


def test_format_silent_when_clean() -> None:
    text = "def foo():\n    return 1\n"
    assert format_edit_diagnostics("a.py", text, old_text="x = 1\n") == ""


def test_write_file_appends_diagnostics(tmp_path: Path) -> None:
    ctx = WorkspaceContext(tmp_path, allow_write=True)
    tool = next(t for t in create_filesystem_tools(ctx) if t.name == "write_file")
    out = str(tool.invoke({"path": "broken.py", "content": "def foo(\n"}))
    assert out.startswith("已写入")
    assert DIAGNOSTIC_MARKER in out
    assert "broken.py:L" in out


def test_write_file_clean_python_has_no_diagnostics(tmp_path: Path) -> None:
    ctx = WorkspaceContext(tmp_path, allow_write=True)
    tool = next(t for t in create_filesystem_tools(ctx) if t.name == "write_file")
    out = str(tool.invoke({"path": "ok.py", "content": "def foo():\n    return 1\n"}))
    assert out.startswith("已写入")
    assert DIAGNOSTIC_MARKER not in out


def test_search_replace_introducing_syntax_error_is_flagged(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("def foo():\n    return 1\n", encoding="utf-8")
    ctx = WorkspaceContext(tmp_path, allow_write=True)
    tool = next(t for t in create_filesystem_tools(ctx) if t.name == "search_replace")
    out = str(
        tool.invoke(
            {
                "path": "sample.py",
                "old_string": "def foo():\n    return 1\n",
                "new_string": "def foo(\n    return 1\n",
            }
        )
    )
    assert out.startswith("已替换")
    assert DIAGNOSTIC_MARKER in out
    assert "sample.py" in out


def test_search_replace_markdown_is_not_diagnosed(tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    target.write_text("# hi\n", encoding="utf-8")
    ctx = WorkspaceContext(tmp_path, allow_write=True)
    tool = next(t for t in create_filesystem_tools(ctx) if t.name == "search_replace")
    out = str(
        tool.invoke(
            {
                "path": "note.md",
                "old_string": "# hi",
                "new_string": "# hello (",
            }
        )
    )
    assert out.startswith("已替换")
    assert DIAGNOSTIC_MARKER not in out


def test_edit_settings_defaults(tmp_path: Path) -> None:
    settings = resolve_edit_settings(tmp_path)
    assert settings.fuzzy_apply is True
    assert settings.syntax_diagnostics is True


def test_edit_settings_can_disable(tmp_path: Path) -> None:
    llgraph_dir = tmp_path / ".llgraph"
    llgraph_dir.mkdir()
    (llgraph_dir / "agent.json").write_text(
        '{"edits": {"fuzzy_apply": false, "syntax_diagnostics": false}}',
        encoding="utf-8",
    )
    settings = resolve_edit_settings(tmp_path)
    assert settings.fuzzy_apply is False
    assert settings.syntax_diagnostics is False

    ctx = WorkspaceContext(tmp_path, allow_write=True)
    write = next(t for t in create_filesystem_tools(ctx) if t.name == "write_file")
    out = str(write.invoke({"path": "broken.py", "content": "def foo(\n"}))
    assert DIAGNOSTIC_MARKER not in out
