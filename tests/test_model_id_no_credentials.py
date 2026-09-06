"""缺网关凭据时：模型名仍可解析，只有真正建客户端才报错。

模型名决定上下文窗口、dispatch profile、Agent 缓存键、trace 展示，
这些只读路径不该因为没配 LLGRAPH_API_KEY 就整片崩。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llgraph.config.config import (
    DEFAULT_MODEL,
    ENV_API_BASE_URL,
    ENV_API_KEY,
    ENV_MODEL,
    get_llgraph_settings,
    resolve_configured_model,
)
from llgraph.core.llm_settings import (
    format_model_banner_suffix,
    format_model_status,
    resolve_effective_model,
    resolve_llm_settings,
    set_runtime_model,
)

WORKSPACE = Path(__file__).resolve().parents[1] / "examples" / "default-workspace"


@pytest.fixture(autouse=True)
def _no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_API_BASE_URL, raising=False)
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    monkeypatch.delenv(ENV_MODEL, raising=False)
    set_runtime_model(None)
    yield
    set_runtime_model(None)


def test_configured_model_falls_back_to_default() -> None:
    assert resolve_configured_model() == DEFAULT_MODEL


def test_configured_model_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_MODEL, "kimi-k2.6")
    assert resolve_configured_model() == "kimi-k2.6"


def test_effective_model_without_credentials() -> None:
    assert resolve_effective_model() == DEFAULT_MODEL
    assert resolve_effective_model(WORKSPACE)


def test_runtime_model_override_without_credentials() -> None:
    set_runtime_model("glm-5")
    assert resolve_effective_model(WORKSPACE) == "glm-5"


def test_llm_settings_and_status_without_credentials() -> None:
    settings = resolve_llm_settings(WORKSPACE)
    assert settings.model
    assert settings.max_tokens >= 1024
    assert "当前模型" in format_model_status(WORKSPACE)
    assert format_model_banner_suffix(WORKSPACE)


def test_credentials_still_required_for_client() -> None:
    with pytest.raises(RuntimeError) as exc:
        get_llgraph_settings()
    message = str(exc.value)
    assert ENV_API_BASE_URL in message
    assert ENV_API_KEY in message

    from llgraph.core.llm import create_gateway_llm

    with pytest.raises(RuntimeError):
        create_gateway_llm(WORKSPACE)
