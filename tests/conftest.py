"""pytest 全局夹具：隔离用户级 ~/.llgraph 与网关凭据，避免单测污染真实目录 / 打真网关。"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from llgraph.config.config import (
    ENV_API_BASE_URL,
    ENV_API_KEY,
    ENV_IGNORE_ENV_FILES,
    ENV_MODEL,
)

# 不可路由的本地端口：任何漏网的出站请求都会立刻 connection refused，不会真打网关
_FAKE_BASE_URL = "http://127.0.0.1:9"
_FAKE_API_KEY = "test-key-not-a-secret"
_FAKE_MODEL = "claude-opus-4-6"


@pytest.fixture(autouse=True)
def _isolate_llgraph_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    每个用例将 LLGRAPH_HOME 指到独立临时目录。

    会话/记忆等用户级落盘均经 ``user_llgraph_home()`` 解析，须运行时读环境变量。
    """
    home = tmp_path_factory.mktemp("llgraph-home")
    monkeypatch.setenv("LLGRAPH_HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def _isolate_gateway_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    用固定假凭据顶掉真实网关配置。

    否则单测结果取决于开发机上有没有 ``~/.config/llgraph/llgraph.env`` 与项目 ``.env``：
    没配就整片报「缺少环境变量」，配了则可能真打网关、烧真 token。
    """
    monkeypatch.setenv(ENV_IGNORE_ENV_FILES, "1")
    monkeypatch.setenv(ENV_API_BASE_URL, _FAKE_BASE_URL)
    monkeypatch.setenv(ENV_API_KEY, _FAKE_API_KEY)
    monkeypatch.setenv(ENV_MODEL, _FAKE_MODEL)


@pytest.fixture(autouse=True)
def _pin_local_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    """本地时间展示按 UTC+8 断言，固定 TZ 免得跑在别的时区上就红。"""
    monkeypatch.setenv("TZ", "Asia/Shanghai")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()
