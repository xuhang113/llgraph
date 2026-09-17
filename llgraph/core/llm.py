"""建 Chat 模型客户端：OpenAI 兼容网关（默认）+ Anthropic / OpenAI / Gemini / Ollama 官方入口。"""

from pathlib import Path
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_anthropic import chat_models as anthropic_chat_models

from llgraph.config.providers import (
    PROVIDER_ANTHROPIC,
    PROVIDER_GATEWAY,
    PROVIDER_GEMINI,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
    resolve_provider_settings,
)
from llgraph.core.llm_settings import LlmSettings, resolve_llm_settings
from llgraph.core.model_thinking import resolve_model_thinking_payload

_USAGE_PATCHED = False

# provider → (pip 包, 可选 extra)
_PROVIDER_PACKAGES: dict[str, tuple[str, str]] = {
    PROVIDER_ANTHROPIC: ("langchain-anthropic", ""),
    PROVIDER_OPENAI: ("langchain-openai", "openai"),
    PROVIDER_GEMINI: ("langchain-google-genai", "gemini"),
    PROVIDER_OLLAMA: ("langchain-ollama", "ollama"),
}


def _missing_package_error(provider: str, exc: ImportError) -> RuntimeError:
    """
    provider 依赖没装时的报错（直接给安装命令）。

    @param provider provider 名
    @param exc 原始 ImportError
    @return RuntimeError
    """
    package, extra = _PROVIDER_PACKAGES.get(provider, ("", ""))
    extra_hint = f"（或 pip install 'llgraph[{extra}]'）" if extra else ""
    return RuntimeError(
        f"provider={provider} 需要 {package}：pip install {package}{extra_hint}。原因: {exc}"
    )


def _patch_gateway_usage_metadata() -> None:
    """
    部分兼容网关的 usage 里 cache 字段可能为 null，
    langchain-anthropic 累加时会 TypeError；此处将 None 视为 0。
    """
    global _USAGE_PATCHED
    if _USAGE_PATCHED:
        return
    original = anthropic_chat_models._create_usage_metadata

    def safe_create_usage_metadata(usage):
        if usage is None:
            return None
        # Gateway 可能返回 cache_creation.*=null，langchain 对 None 做 += 会报错
        cache_creation = getattr(usage, "cache_creation", None)
        if cache_creation is not None:
            for attr in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"):
                if hasattr(cache_creation, attr) and getattr(cache_creation, attr) is None:
                    setattr(cache_creation, attr, 0)
        for attr in ("cache_creation_input_tokens", "cache_read_input_tokens"):
            if hasattr(usage, attr) and getattr(usage, attr) is None:
                setattr(usage, attr, 0)
        return original(usage)

    anthropic_chat_models._create_usage_metadata = safe_create_usage_metadata
    _USAGE_PATCHED = True


def _thinking_request(workspace: Path | None, model: str) -> tuple[dict | None, str | None]:
    """
    解析 thinking 请求体与 effort。

    @param workspace 工作区根
    @param model 模型 id
    @return (thinking 字段, effort)
    """
    from llgraph.core.model_thinking import split_thinking_payload

    return split_thinking_payload(resolve_model_thinking_payload(workspace, model))


def _create_anthropic_chat(
    *,
    llm_cfg: LlmSettings,
    api_key: str,
    base_url: str,
    workspace: Path | None,
    gateway: bool,
) -> ChatAnthropic:
    """
    建 Anthropic 协议客户端（官方 API 与 OpenAI 兼容网关共用）。

    @param llm_cfg 生成参数
    @param api_key API Key
    @param base_url 网关地址；官方入口留空走 SDK 默认
    @param workspace 工作区根
    @param gateway 是否走兼容网关（决定要不要打网关补丁）
    @return ChatAnthropic
    """
    if gateway:
        _patch_gateway_usage_metadata()
        from llgraph.core.gateway_kimi_patch import patch_gateway_kimi_reasoning_payload

        patch_gateway_kimi_reasoning_payload()

    thinking_body, thinking_effort = _thinking_request(workspace, llm_cfg.model)
    # 不显式传 temperature：部分网关对 temperature=0.2 会返回 400
    llm_kwargs: dict[str, Any] = {
        "model": llm_cfg.model,
        "api_key": api_key,
        "max_tokens": llm_cfg.max_tokens,
    }
    if base_url:
        llm_kwargs["base_url"] = base_url
    if thinking_body is not None:
        llm_kwargs["thinking"] = thinking_body
    if thinking_effort:
        # Claude adaptive：effort 必须在 output_config，langchain 用顶层 effort 参数
        llm_kwargs["effort"] = thinking_effort
    # 单次模型 HTTP 超时，避免网关无响应时 Web 会话永久占锁
    llm_kwargs["timeout"] = float(llm_cfg.request_timeout_sec)
    return ChatAnthropic(**llm_kwargs)


def _create_openai_chat(
    *,
    llm_cfg: LlmSettings,
    api_key: str,
    base_url: str,
    workspace: Path | None,
) -> Any:
    """
    建 OpenAI 官方入口客户端。

    @param llm_cfg 生成参数
    @param api_key API Key
    @param base_url 自定义 base_url（留空用官方）
    @param workspace 工作区根
    @return ChatOpenAI
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise _missing_package_error(PROVIDER_OPENAI, exc) from exc

    _thinking_body, thinking_effort = _thinking_request(workspace, llm_cfg.model)
    llm_kwargs: dict[str, Any] = {
        "model": llm_cfg.model,
        "api_key": api_key,
        "max_tokens": llm_cfg.max_tokens,
        "timeout": float(llm_cfg.request_timeout_sec),
    }
    if base_url:
        llm_kwargs["base_url"] = base_url
    # Anthropic 的 thinking body 在 OpenAI 侧没有对应字段，只有 effort 能映射；
    # 非推理模型传 reasoning_effort 会 400，所以只在配置里点名了 effort 时才带。
    if thinking_effort:
        llm_kwargs["reasoning_effort"] = thinking_effort
    return ChatOpenAI(**llm_kwargs)


def _create_gemini_chat(*, llm_cfg: LlmSettings, api_key: str) -> Any:
    """
    建 Gemini 官方入口客户端。

    @param llm_cfg 生成参数
    @param api_key API Key
    @return ChatGoogleGenerativeAI
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise _missing_package_error(PROVIDER_GEMINI, exc) from exc

    return ChatGoogleGenerativeAI(
        model=llm_cfg.model,
        google_api_key=api_key,
        max_output_tokens=llm_cfg.max_tokens,
        timeout=float(llm_cfg.request_timeout_sec),
    )


def _create_ollama_chat(*, llm_cfg: LlmSettings, base_url: str) -> Any:
    """
    建本地 Ollama 客户端（无需 API Key）。

    @param llm_cfg 生成参数
    @param base_url ollama 地址
    @return ChatOllama
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise _missing_package_error(PROVIDER_OLLAMA, exc) from exc

    return ChatOllama(
        model=llm_cfg.model,
        base_url=base_url,
        num_predict=llm_cfg.max_tokens,
        client_kwargs={"timeout": float(llm_cfg.request_timeout_sec)},
    )


def create_chat_llm(workspace: Path | None = None) -> Any:
    """
    按当前 provider 创建 Chat 模型实例。

    provider 与凭据解析见 ``llgraph/config/providers.py``；默认仍是 OpenAI 兼容网关
    （``LLGRAPH_API_BASE_URL`` + ``LLGRAPH_API_KEY``）。
    max_tokens 来自工作区 .llgraph/agent.json → llm.max_tokens（默认 16384）。

    @param workspace 工作区根，用于读取 agent.json
    @return LangChain Chat 模型
    """
    ws = Path(workspace).expanduser().resolve() if workspace is not None else None
    llm_cfg = resolve_llm_settings(ws)
    provider_cfg = resolve_provider_settings(
        provider=llm_cfg.provider,
        model_id=llm_cfg.model,
    )
    provider = provider_cfg.provider

    if provider in (PROVIDER_GATEWAY, PROVIDER_ANTHROPIC):
        llm = _create_anthropic_chat(
            llm_cfg=llm_cfg,
            api_key=provider_cfg.api_key,
            base_url=provider_cfg.base_url,
            workspace=ws,
            gateway=provider == PROVIDER_GATEWAY,
        )
    elif provider == PROVIDER_OPENAI:
        llm = _create_openai_chat(
            llm_cfg=llm_cfg,
            api_key=provider_cfg.api_key,
            base_url=provider_cfg.base_url,
            workspace=ws,
        )
    elif provider == PROVIDER_GEMINI:
        llm = _create_gemini_chat(llm_cfg=llm_cfg, api_key=provider_cfg.api_key)
    elif provider == PROVIDER_OLLAMA:
        llm = _create_ollama_chat(llm_cfg=llm_cfg, base_url=provider_cfg.base_url)
    else:  # 理论不可达：providers 已归一
        raise RuntimeError(f"未支持的模型入口: {provider}")

    # 供 gateway reasoning 注入读取 agent.json dispatch profile
    object.__setattr__(llm, "llgraph_workspace", ws)
    # 供 bind_tools / prompt cache 判断能传哪些厂商专属参数
    object.__setattr__(llm, "llgraph_provider", provider)
    return llm


def create_gateway_llm(workspace: Path | None = None) -> Any:
    """
    旧名，等价于 create_chat_llm（历史上只有网关一条路）。

    @param workspace 工作区根
    @return LangChain Chat 模型
    """
    return create_chat_llm(workspace)


def verify_model_credentials(workspace: Path | None = None) -> str:
    """
    启动前检查模型入口凭据（缺失时抛 RuntimeError），不建客户端。

    @param workspace 工作区根
    @return 选中的 provider
    """
    ws = Path(workspace).expanduser().resolve() if workspace is not None else None
    llm_cfg = resolve_llm_settings(ws)
    return resolve_provider_settings(
        provider=llm_cfg.provider,
        model_id=llm_cfg.model,
    ).provider


__all__ = [
    "create_chat_llm",
    "create_gateway_llm",
    "verify_model_credentials",
]
