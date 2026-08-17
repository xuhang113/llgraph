"""pytest 全局夹具：隔离用户级 ~/.llgraph，避免单测污染真实目录。"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_llgraph_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    每个用例将 LLGRAPH_HOME 指到独立临时目录。

    会话/记忆等用户级落盘均经 ``user_llgraph_home()`` 解析，须运行时读环境变量。
    """
    home = tmp_path_factory.mktemp("llgraph-home")
    monkeypatch.setenv("LLGRAPH_HOME", str(home))
    return home
