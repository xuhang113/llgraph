"""工作区列表过滤：examples 模板目录。"""

from __future__ import annotations

from pathlib import Path

from llgraph.config.workspace_config import (
    is_packaged_example_workspace,
    package_user_config_dir,
)


def test_is_packaged_example_workspace_user_llgraph() -> None:
    path = package_user_config_dir()
    assert path.is_dir() or path.name == "user-llgraph"
    assert is_packaged_example_workspace(path)


def test_is_packaged_example_workspace_normal_project(tmp_path: Path) -> None:
    project = tmp_path / "WorkspaceV2"
    project.mkdir()
    assert not is_packaged_example_workspace(project)
