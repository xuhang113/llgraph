"""路径容错：唯一解析与相近路径（对标 Cursor / Claude Code）。"""

from __future__ import annotations

from pathlib import Path

from llgraph.core.filesystem_tools import create_filesystem_tools
from llgraph.core.path_recover import (
    AUTO_RESOLVE_MARKER,
    extract_glob_literal_name,
    invalidate_path_listing_cache,
    resolve_tool_path,
    suggest_paths,
)
from llgraph.core.workspace import WorkspaceContext


def _layout(root: Path) -> Path:
    pkg = root / "pkg" / "core"
    pkg.mkdir(parents=True)
    (pkg / "agent.py").write_text("AGENT = 1\n", encoding="utf-8")
    (pkg / "filesystem_tools.py").write_text("FS = 1\n", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "readme.md").write_text("# docs\n", encoding="utf-8")
    invalidate_path_listing_cache(root)
    return root


def _tools(root: Path, *, allow_write: bool = False):
    ctx = WorkspaceContext(root, allow_write=allow_write)
    return {tool.name: tool for tool in create_filesystem_tools(ctx)}


def test_extract_glob_literal_name() -> None:
    assert extract_glob_literal_name("**/FooService.java") == "FooService.java"
    assert extract_glob_literal_name("FooService.java") == "FooService.java"
    assert extract_glob_literal_name("**/*.java") == ""
    assert extract_glob_literal_name("**/*Foo.java") == ""


def test_unique_suffix_auto_resolves_read(tmp_path: Path) -> None:
    root = _layout(tmp_path)
    tools = _tools(root)
    out = tools["read_file"].invoke({"path": "core/agent.py"})
    assert AUTO_RESOLVE_MARKER in out
    assert "pkg/core/agent.py" in out
    assert "AGENT = 1" in out


def test_unique_basename_auto_resolves_read(tmp_path: Path) -> None:
    root = _layout(tmp_path)
    tools = _tools(root)
    out = tools["read_file"].invoke({"path": "filesystem_tools.py"})
    assert AUTO_RESOLVE_MARKER in out
    assert "pkg/core/filesystem_tools.py" in out
    assert "FS = 1" in out


def test_ambiguous_basename_suggests_not_auto(tmp_path: Path) -> None:
    root = _layout(tmp_path)
    (root / "other").mkdir()
    (root / "other" / "agent.py").write_text("OTHER = 1\n", encoding="utf-8")
    invalidate_path_listing_cache(root)
    tools = _tools(root)
    out = tools["read_file"].invoke({"path": "agent.py"})
    assert AUTO_RESOLVE_MARKER not in out
    assert "文件不存在" in out
    assert "pkg/core/agent.py" in out
    assert "other/agent.py" in out
    assert "AGENT = 1" not in out


def test_unique_suffix_still_auto_when_basename_ambiguous(tmp_path: Path) -> None:
    root = _layout(tmp_path)
    (root / "other").mkdir()
    (root / "other" / "agent.py").write_text("OTHER = 1\n", encoding="utf-8")
    invalidate_path_listing_cache(root)
    tools = _tools(root)
    out = tools["read_file"].invoke({"path": "core/agent.py"})
    assert AUTO_RESOLVE_MARKER in out
    assert "pkg/core/agent.py" in out
    assert "AGENT = 1" in out
    assert "OTHER = 1" not in out


def test_workspace_name_prefix_stripped(tmp_path: Path) -> None:
    ws = tmp_path / "demoapp"
    ws.mkdir()
    (ws / "src").mkdir()
    (ws / "src" / "main.py").write_text("MAIN = True\n", encoding="utf-8")
    invalidate_path_listing_cache(ws)
    tools = _tools(ws)
    out = tools["read_file"].invoke({"path": "demoapp/src/main.py"})
    assert AUTO_RESOLVE_MARKER in out
    assert "src/main.py" in out
    assert "MAIN = True" in out


def test_read_directory_lists_entries(tmp_path: Path) -> None:
    root = _layout(tmp_path)
    tools = _tools(root)
    out = tools["read_file"].invoke({"path": "pkg/core"})
    assert "这是目录" in out
    assert "pkg/core/agent.py" in out
    assert "AGENT = 1" not in out


def test_list_directory_auto_resolves_unique_dir(tmp_path: Path) -> None:
    root = _layout(tmp_path)
    tools = _tools(root)
    out = tools["list_directory"].invoke({"path": "core"})
    assert AUTO_RESOLVE_MARKER in out
    assert "pkg/core/agent.py" in out


def test_list_directory_on_file_shows_parent(tmp_path: Path) -> None:
    root = _layout(tmp_path)
    tools = _tools(root)
    out = tools["list_directory"].invoke({"path": "pkg/core/agent.py"})
    assert "不是目录" in out
    assert "read_file" in out
    assert "pkg/core/agent.py" in out


def test_missing_path_with_no_neighbors(tmp_path: Path) -> None:
    root = _layout(tmp_path)
    tools = _tools(root)
    out = tools["read_file"].invoke({"path": "zzz_no_such_file_abc.py"})
    assert "文件不存在" in out
    assert AUTO_RESOLVE_MARKER not in out


def test_search_replace_auto_resolves_unique_path(tmp_path: Path) -> None:
    root = _layout(tmp_path)
    tools = _tools(root, allow_write=True)
    out = tools["search_replace"].invoke(
        {
            "path": "core/agent.py",
            "old_string": "AGENT = 1",
            "new_string": "AGENT = 2",
        }
    )
    assert AUTO_RESOLVE_MARKER in out
    assert "pkg/core/agent.py" in out
    assert "已替换" in out or "AGENT = 2" in out
    assert (root / "pkg" / "core" / "agent.py").read_text(encoding="utf-8") == "AGENT = 2\n"


def test_search_replace_ambiguous_does_not_write(tmp_path: Path) -> None:
    root = _layout(tmp_path)
    (root / "other").mkdir()
    (root / "other" / "agent.py").write_text("AGENT = 1\n", encoding="utf-8")
    invalidate_path_listing_cache(root)
    tools = _tools(root, allow_write=True)
    out = tools["search_replace"].invoke(
        {
            "path": "agent.py",
            "old_string": "AGENT = 1",
            "new_string": "AGENT = 9",
        }
    )
    assert "文件不存在" in out
    assert "pkg/core/agent.py" in out
    assert (root / "pkg" / "core" / "agent.py").read_text(encoding="utf-8") == "AGENT = 1\n"
    assert (root / "other" / "agent.py").read_text(encoding="utf-8") == "AGENT = 1\n"


def test_write_file_does_not_remap_new_path(tmp_path: Path) -> None:
    root = _layout(tmp_path)
    tools = _tools(root, allow_write=True)
    out = tools["write_file"].invoke(
        {"path": "core/new_module.py", "content": "NEW = 1\n"}
    )
    assert "已写入" in out
    assert (root / "core" / "new_module.py").is_file()
    assert not (root / "pkg" / "core" / "new_module.py").exists()
    assert (root / "pkg" / "core" / "agent.py").read_text(encoding="utf-8") == "AGENT = 1\n"


def test_append_file_auto_resolves_existing(tmp_path: Path) -> None:
    root = _layout(tmp_path)
    tools = _tools(root, allow_write=True)
    out = tools["append_file"].invoke(
        {"path": "filesystem_tools.py", "content": "FS2 = 2\n"}
    )
    assert AUTO_RESOLVE_MARKER in out
    text = (root / "pkg" / "core" / "filesystem_tools.py").read_text(encoding="utf-8")
    assert "FS2 = 2" in text


def test_grep_auto_resolves_unique_dir(tmp_path: Path, monkeypatch) -> None:
    class _Ctx:
        grep_context_lines = 2

    monkeypatch.setattr(
        "llgraph.core.filesystem_tools.ripgrep_available",
        lambda: False,
    )
    monkeypatch.setattr(
        "llgraph.context.context_settings.resolve_context_settings",
        lambda _root: _Ctx(),
    )
    root = _layout(tmp_path)
    tools = _tools(root)
    out = tools["grep_files"].invoke({"pattern": "AGENT", "path": "core"})
    assert AUTO_RESOLVE_MARKER in out
    assert "pkg/core/agent.py" in out
    assert "AGENT" in out


def test_grep_missing_path_suggests(tmp_path: Path) -> None:
    root = _layout(tmp_path)
    tools = _tools(root)
    out = tools["grep_files"].invoke({"pattern": "AGENT", "path": "no_such_dir_xyz"})
    assert "路径不存在" in out
    assert AUTO_RESOLVE_MARKER not in out


def test_suggest_paths_ranks_suffix_first(tmp_path: Path) -> None:
    root = _layout(tmp_path)
    (root / "other").mkdir()
    (root / "other" / "agent.py").write_text("OTHER = 1\n", encoding="utf-8")
    invalidate_path_listing_cache(root)
    ctx = WorkspaceContext(root)
    hits = suggest_paths(ctx, "core/agent.py", want="file")
    assert hits
    assert hits[0].rel == "pkg/core/agent.py"
    assert hits[0].reason == "suffix"
    resolved = resolve_tool_path(ctx, "core/agent.py", want="file")
    assert resolved.ok
    assert resolved.kind == "auto"
    assert resolved.rel == "pkg/core/agent.py"


def test_glob_literal_unique_when_path_missing(tmp_path: Path) -> None:
    root = _layout(tmp_path)
    tools = _tools(root)
    out = tools["glob_files"].invoke(
        {"glob_pattern": "**/filesystem_tools.py", "path": "wrong/pkg"}
    )
    assert AUTO_RESOLVE_MARKER in out
    assert "filesystem_tools.py" in out
    assert "pkg/core/filesystem_tools.py" in out
