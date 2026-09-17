"""模型入口：网关（默认）+ Anthropic / OpenAI / Gemini / Ollama 官方入口。

回归重点是「不能坏的那条」：网关凭据在时，无论机器上还 export 了谁家的官方 Key，
选中的都必须还是网关，请求参数与多 provider 之前一致。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llgraph.config.config import (
    DEFAULT_MODEL,
    ENV_API_BASE_URL,
    ENV_API_KEY,
    ENV_MODEL,
)
from llgraph.config.providers import (
    ENV_PROVIDER,
    PROVIDER_ANTHROPIC,
    PROVIDER_GATEWAY,
    PROVIDER_GEMINI,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
    default_model_for_provider,
    detect_provider,
    normalize_ollama_base_url,
    normalize_provider_name,
    provider_from_model_id,
    provider_supports_prompt_cache,
    resolve_provider_settings,
    tool_bind_kwargs,
)
from llgraph.core.llm import create_chat_llm, verify_model_credentials
from llgraph.core.llm_settings import (
    resolve_effective_model,
    resolve_effective_provider,
    set_runtime_model,
)
from llgraph.core.prompt_cache_settings import prompt_cache_enabled_for_model

WORKSPACE = Path(__file__).resolve().parents[1] / "examples" / "default-workspace"

_FAKE_KEY = "test-key-not-a-secret"


@pytest.fixture(autouse=True)
def _clear_runtime_model() -> None:
    set_runtime_model(None)
    yield
    set_runtime_model(None)


def _drop_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    """去掉 conftest 注入的网关假凭据（测其他 provider 时用）。"""
    monkeypatch.delenv(ENV_API_BASE_URL, raising=False)
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    monkeypatch.delenv(ENV_MODEL, raising=False)


# --- provider 选择 ---


def test_gateway_wins_when_credentials_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """网关凭据在 + 机器上还有官方 Key：仍走网关（老路径不能被抢走）。"""
    monkeypatch.setenv("ANTHROPIC_API_KEY", _FAKE_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", _FAKE_KEY)
    provider, source = detect_provider("claude-opus-4-6")
    assert provider == PROVIDER_GATEWAY
    assert ENV_API_BASE_URL in source


def test_explicit_provider_beats_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_PROVIDER, "Claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", _FAKE_KEY)
    provider, source = detect_provider("claude-sonnet-4-5")
    assert provider == PROVIDER_ANTHROPIC
    assert ENV_PROVIDER in source


def test_unknown_provider_name_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_PROVIDER, "deepmind")
    with pytest.raises(RuntimeError) as exc:
        detect_provider("gemini-2.5-flash")
    assert "deepmind" in str(exc.value)


@pytest.mark.parametrize(
    ("env_name", "expected"),
    [
        ("ANTHROPIC_API_KEY", PROVIDER_ANTHROPIC),
        ("LLGRAPH_ANTHROPIC_API_KEY", PROVIDER_ANTHROPIC),
        ("OPENAI_API_KEY", PROVIDER_OPENAI),
        ("LLGRAPH_OPENAI_API_KEY", PROVIDER_OPENAI),
        ("GEMINI_API_KEY", PROVIDER_GEMINI),
        ("GOOGLE_API_KEY", PROVIDER_GEMINI),
    ],
)
def test_single_vendor_key_selects_provider(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    expected: str,
) -> None:
    """只 export 了一家的官方 Key：开箱就走那家，且用那家的默认模型。"""
    _drop_gateway(monkeypatch)
    monkeypatch.setenv(env_name, _FAKE_KEY)
    provider, _source = detect_provider(None)
    assert provider == expected
    assert resolve_effective_model() == default_model_for_provider(expected)


def test_model_id_picks_provider_when_several_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """两家 Key 都在时，按模型 id 走（/model 切模型即切入口）。"""
    _drop_gateway(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", _FAKE_KEY)
    monkeypatch.setenv("GEMINI_API_KEY", _FAKE_KEY)
    monkeypatch.setenv(ENV_MODEL, "gemini-2.5-pro")
    provider, source = detect_provider("gemini-2.5-pro")
    assert provider == PROVIDER_GEMINI
    assert "gemini-2.5-pro" in source


def test_ollama_from_base_url_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """本地 Ollama 不需要 Key；OLLAMA_HOST 常写成没 scheme 的样子。"""
    _drop_gateway(monkeypatch)
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
    provider, _source = detect_provider(None)
    assert provider == PROVIDER_OLLAMA
    settings = resolve_provider_settings(provider=None, model_id=None)
    assert settings.base_url == "http://127.0.0.1:11434"
    assert settings.api_key == ""
    assert resolve_effective_model() == "qwen3:8b"


def test_no_credentials_lists_all_four_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """一家都没配：报错要把四条路一起说，不能只提网关。"""
    _drop_gateway(monkeypatch)
    assert detect_provider(None) == (None, "")
    # 只读路径不许崩：模型名与 provider 仍可解析
    assert resolve_effective_model() == DEFAULT_MODEL
    assert resolve_effective_provider()[0] == PROVIDER_GATEWAY

    with pytest.raises(RuntimeError) as exc:
        resolve_provider_settings(provider=None, model_id=None)
    message = str(exc.value)
    for expected in (ENV_API_BASE_URL, "ANTHROPIC_API_KEY", "ollama", ENV_PROVIDER):
        assert expected in message


def test_provider_selected_but_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _drop_gateway(monkeypatch)
    monkeypatch.setenv(ENV_PROVIDER, "openai")
    with pytest.raises(RuntimeError) as exc:
        resolve_provider_settings(provider=None, model_id=None)
    assert "OPENAI_API_KEY" in str(exc.value)


def test_workspace_agent_json_provider_wins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """agent.json llm.provider 压过环境变量探测（网关凭据在也照样切）。"""
    (tmp_path / ".llgraph").mkdir()
    (tmp_path / ".llgraph" / "agent.json").write_text(
        '{"llm": {"provider": "ollama", "model": "qwen3:8b"}}',
        encoding="utf-8",
    )
    provider, source = resolve_effective_provider(tmp_path)
    assert provider == PROVIDER_OLLAMA
    assert "agent.json" in source
    assert resolve_effective_model(tmp_path) == "qwen3:8b"


def test_provider_name_aliases_and_model_guess() -> None:
    assert normalize_provider_name("Anthropic ") == PROVIDER_ANTHROPIC
    assert normalize_provider_name("google-genai") == PROVIDER_GEMINI
    assert normalize_provider_name("openai_compatible") == PROVIDER_GATEWAY
    assert normalize_provider_name("bedrock") is None
    assert normalize_provider_name(None) is None

    assert provider_from_model_id("claude-opus-4-6") == PROVIDER_ANTHROPIC
    assert provider_from_model_id("gpt-4.1") == PROVIDER_OPENAI
    assert provider_from_model_id("o3-mini") == PROVIDER_OPENAI
    assert provider_from_model_id("models/gemini-2.5-flash") == PROVIDER_GEMINI
    assert provider_from_model_id("llama3.1:8b") == PROVIDER_OLLAMA
    assert provider_from_model_id("kimi-k2.6") is None
    assert normalize_ollama_base_url("") .endswith(":11434")


# --- 厂商专属参数只发给认它的那家 ---


def test_tool_bind_kwargs_by_provider() -> None:
    for provider in (PROVIDER_GATEWAY, PROVIDER_ANTHROPIC, PROVIDER_OPENAI):
        assert tool_bind_kwargs(provider) == {
            "parallel_tool_calls": True,
            "tool_choice": "auto",
        }
    assert tool_bind_kwargs(PROVIDER_GEMINI) == {}
    assert tool_bind_kwargs(PROVIDER_OLLAMA) == {}
    # 没挂 provider 的模型按老行为（网关）
    assert tool_bind_kwargs(None)["parallel_tool_calls"] is True


def test_bind_tools_passes_only_supported_kwargs() -> None:
    """_bind_tools_if_needed 按模型上挂的 provider 决定传什么。"""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool

    from llgraph.core.react_graph import _bind_tools_if_needed

    @tool
    def ping(city: str) -> str:
        """查一个城市。"""
        return city

    seen: list[dict[str, Any]] = []

    class _Recorder(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):  # type: ignore[no-untyped-def]
            seen.append(dict(kwargs))
            return self

    model = _Recorder(responses=[AIMessage(content="ok")])
    object.__setattr__(model, "llgraph_provider", PROVIDER_OLLAMA)
    _bind_tools_if_needed(model, [ping])
    assert seen == [{}]

    seen.clear()
    object.__setattr__(model, "llgraph_provider", PROVIDER_OPENAI)
    _bind_tools_if_needed(model, [ping])
    assert seen == [{"parallel_tool_calls": True, "tool_choice": "auto"}]


def test_prompt_cache_only_for_anthropic_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    """cache_control 是 Anthropic 的字段，别打到 OpenAI / Gemini / Ollama 的请求里。"""
    assert prompt_cache_enabled_for_model(WORKSPACE, "claude-opus-4-6") is True

    _drop_gateway(monkeypatch)
    monkeypatch.setenv(ENV_PROVIDER, "ollama")
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
    assert prompt_cache_enabled_for_model(WORKSPACE, "qwen3:8b") is False

    monkeypatch.setenv(ENV_PROVIDER, "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", _FAKE_KEY)
    assert prompt_cache_enabled_for_model(WORKSPACE, "claude-sonnet-4-5") is True


def test_provider_supports_prompt_cache_table() -> None:
    assert provider_supports_prompt_cache(PROVIDER_GATEWAY) is True
    assert provider_supports_prompt_cache(PROVIDER_ANTHROPIC) is True
    assert provider_supports_prompt_cache(PROVIDER_OPENAI) is False
    assert provider_supports_prompt_cache(None) is True


# --- 工厂建出来的客户端 ---


def test_gateway_client_unchanged() -> None:
    """网关路径：仍是 ChatAnthropic + base_url 指向网关。"""
    from langchain_anthropic import ChatAnthropic

    llm = create_chat_llm(WORKSPACE)
    assert isinstance(llm, ChatAnthropic)
    assert str(llm.anthropic_api_url).startswith("http://127.0.0.1:9")
    assert getattr(llm, "llgraph_provider") == PROVIDER_GATEWAY
    assert verify_model_credentials(WORKSPACE) == PROVIDER_GATEWAY


def test_anthropic_official_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """官方 Anthropic：不传 base_url（走 SDK 默认），也不打网关补丁。"""
    from langchain_anthropic import ChatAnthropic

    _drop_gateway(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _FAKE_KEY)
    # 不带工作区：example workspace 的 agent.json 点名了模型，会盖掉 provider 默认值
    llm = create_chat_llm(None)
    assert isinstance(llm, ChatAnthropic)
    assert llm.model == "claude-sonnet-4-5"
    assert "127.0.0.1:9" not in str(llm.anthropic_api_url)
    assert getattr(llm, "llgraph_provider") == PROVIDER_ANTHROPIC


def test_openai_official_client(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("langchain_openai")
    from langchain_openai import ChatOpenAI

    _drop_gateway(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", _FAKE_KEY)
    monkeypatch.setenv(ENV_MODEL, "gpt-4.1-mini")
    llm = create_chat_llm(None)
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "gpt-4.1-mini"
    assert getattr(llm, "llgraph_provider") == PROVIDER_OPENAI


def test_gemini_official_client(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("langchain_google_genai")
    from langchain_google_genai import ChatGoogleGenerativeAI

    _drop_gateway(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", _FAKE_KEY)
    llm = create_chat_llm(None)
    assert isinstance(llm, ChatGoogleGenerativeAI)
    assert llm.model.endswith("gemini-2.5-flash")
    assert getattr(llm, "llgraph_provider") == PROVIDER_GEMINI


def test_ollama_client(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("langchain_ollama")
    from langchain_ollama import ChatOllama

    _drop_gateway(monkeypatch)
    monkeypatch.setenv("LLGRAPH_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv(ENV_MODEL, "qwen3:8b")
    llm = create_chat_llm(None)
    assert isinstance(llm, ChatOllama)
    assert llm.model == "qwen3:8b"
    assert llm.base_url == "http://127.0.0.1:11434"
    assert getattr(llm, "llgraph_provider") == PROVIDER_OLLAMA


def test_local_tag_model_gets_small_context_window() -> None:
    """本地模型（qwen3:8b）不能按云端 200K 算，否则自动压缩永不触发。"""
    from llgraph.core.model_context_window import (
        LOCAL_TAG_CONTEXT_WINDOW,
        resolve_model_context_window,
    )

    window, source = resolve_model_context_window(None, "qwen3:8b")
    assert window == LOCAL_TAG_CONTEXT_WINDOW
    assert source.startswith("local-tag")
    # 云端模型不受影响
    assert resolve_model_context_window(None, "claude-opus-4-6")[0] == 200_000


def test_max_tokens_and_timeout_follow_agent_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """agent.json 的 llm.max_tokens / request_timeout_sec 对各家都生效。"""
    pytest.importorskip("langchain_openai")
    _drop_gateway(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", _FAKE_KEY)
    monkeypatch.setenv(ENV_PROVIDER, "openai")
    from llgraph.core.llm_settings import resolve_llm_settings

    expected = resolve_llm_settings(WORKSPACE)
    llm = create_chat_llm(WORKSPACE)
    assert llm.max_tokens == expected.max_tokens
    assert float(llm.request_timeout) == expected.request_timeout_sec
