"""模型入口 provider 解析：OpenAI 兼容网关 / Anthropic / OpenAI / Gemini / Ollama。

只回答两个问题：这轮走哪家、凭据读哪个环境变量。不 import 任何 langchain provider 包，
也不读工作区 agent.json（那层在 ``core/llm_settings.py``）。

现有 OpenAI 兼容网关（``LLGRAPH_API_BASE_URL`` + ``LLGRAPH_API_KEY``）仍是默认路径：
只要这两个变量齐，没有显式 ``LLGRAPH_PROVIDER`` 时一律走网关，与本模块加入前一致。
"""

from __future__ import annotations

import os
import re
import socket
import time
from dataclasses import dataclass
from urllib.parse import urlparse

from llgraph.config.config import (
    ENV_API_BASE_URL,
    ENV_API_KEY,
    gateway_credentials_present,
    get_llgraph_settings,
    load_llgraph_env,
)

PROVIDER_GATEWAY = "gateway"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"
PROVIDER_OLLAMA = "ollama"

KNOWN_PROVIDERS: tuple[str, ...] = (
    PROVIDER_GATEWAY,
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI,
    PROVIDER_GEMINI,
    PROVIDER_OLLAMA,
)

ENV_PROVIDER = "LLGRAPH_PROVIDER"
# 置 0 时不去探本机 ollama 端口（单测隔离 / 不想让空配置自动落到本地模型）
ENV_OLLAMA_AUTODETECT = "LLGRAPH_OLLAMA_AUTODETECT"

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# ollama 端口探测结果缓存（地址 → (可连, 时刻)）
_REACHABLE_TTL_SEC = 30.0
_reachable_cache: dict[str, tuple[bool, float]] = {}

# 写法宽松些：claude / google / local 这些叫法都认
_PROVIDER_ALIASES: dict[str, str] = {
    "gateway": PROVIDER_GATEWAY,
    "llgraph": PROVIDER_GATEWAY,
    "openai-compatible": PROVIDER_GATEWAY,
    "openai_compatible": PROVIDER_GATEWAY,
    "compat": PROVIDER_GATEWAY,
    "anthropic": PROVIDER_ANTHROPIC,
    "claude": PROVIDER_ANTHROPIC,
    "openai": PROVIDER_OPENAI,
    "gpt": PROVIDER_OPENAI,
    "gemini": PROVIDER_GEMINI,
    "google": PROVIDER_GEMINI,
    "google-genai": PROVIDER_GEMINI,
    "googleai": PROVIDER_GEMINI,
    "ollama": PROVIDER_OLLAMA,
    "local": PROVIDER_OLLAMA,
}

# API Key：先 LLGRAPH_ 前缀（llgraph 自己的），再官方变量名（开箱即用）
_KEY_ENVS: dict[str, tuple[str, ...]] = {
    PROVIDER_ANTHROPIC: ("LLGRAPH_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    PROVIDER_OPENAI: ("LLGRAPH_OPENAI_API_KEY", "OPENAI_API_KEY"),
    PROVIDER_GEMINI: (
        "LLGRAPH_GEMINI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ),
    PROVIDER_OLLAMA: (),
}

# base_url 只认 LLGRAPH_ 前缀：官方 *_BASE_URL 常被别的工具指到自建代理上，
# 悄悄拿来用会让「我配的是官方 Key」和「请求实际去哪」对不上。
# OLLAMA_HOST 是例外，它就是 ollama 自己的地址变量。
_BASE_URL_ENVS: dict[str, tuple[str, ...]] = {
    PROVIDER_ANTHROPIC: ("LLGRAPH_ANTHROPIC_BASE_URL",),
    PROVIDER_OPENAI: ("LLGRAPH_OPENAI_BASE_URL",),
    PROVIDER_GEMINI: (),
    PROVIDER_OLLAMA: ("LLGRAPH_OLLAMA_BASE_URL", "OLLAMA_HOST"),
}

# 用户没点名模型时，这家自己的默认模型
_DEFAULT_MODELS: dict[str, str] = {
    PROVIDER_ANTHROPIC: "claude-sonnet-4-5",
    PROVIDER_OPENAI: "gpt-4.1",
    PROVIDER_GEMINI: "gemini-2.5-flash",
    PROVIDER_OLLAMA: "qwen3:8b",
}

# 模型 id → provider（仅在没有显式 provider、也没有网关凭据时用）
_MODEL_PATTERNS: tuple[tuple[str, str], ...] = (
    (PROVIDER_ANTHROPIC, r"^(anthropic/)?claude[-.]"),
    (PROVIDER_OPENAI, r"^(openai/)?(gpt[-.]|o[1-9]([-.]|$)|chatgpt)"),
    (PROVIDER_GEMINI, r"^(google/|models/)?gemini[-.]"),
    (PROVIDER_OLLAMA, r"^ollama/"),
)

_KEY_ENV_HINT = ", ".join(
    f"{provider}: {_KEY_ENVS[provider][0]}"
    for provider in (PROVIDER_ANTHROPIC, PROVIDER_OPENAI, PROVIDER_GEMINI)
)


@dataclass(frozen=True)
class ProviderSettings:
    """一次模型调用的入口配置。"""

    provider: str
    api_key: str
    base_url: str
    source: str


def normalize_provider_name(raw: object) -> str | None:
    """
    归一 provider 名（大小写、别名）。

    @param raw 配置里写的值
    @return 规范名；无法识别时 None
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    key = raw.strip().lower().replace(" ", "")
    return _PROVIDER_ALIASES.get(key)


def provider_from_model_id(model_id: object) -> str | None:
    """
    按模型 id 猜 provider。

    ``llama3.1:8b`` 这类带 ``:`` 标签的写法是 ollama 约定，官方三家都不用。

    @param model_id 模型 id
    @return provider 名；猜不出时 None
    """
    if not isinstance(model_id, str) or not model_id.strip():
        return None
    mid = model_id.strip().lower()
    for provider, pattern in _MODEL_PATTERNS:
        if re.search(pattern, mid):
            return provider
    if ":" in mid:
        return PROVIDER_OLLAMA
    return None


def default_model_for_provider(provider: str) -> str | None:
    """
    这家的默认模型（用户没配 LLGRAPH_MODEL / agent.json llm.model 时用）。

    @param provider provider 名
    @return 模型 id；网关沿用 DEFAULT_MODEL 因此返回 None
    """
    return _DEFAULT_MODELS.get(provider)


def _env_first(names: tuple[str, ...]) -> tuple[str, str]:
    """
    取第一个非空环境变量。

    @param names 候选变量名（按优先级）
    @return (值, 命中的变量名)；都为空时 ("", "")
    """
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value, name
    return "", ""


def normalize_ollama_base_url(raw: str) -> str:
    """
    补全 ollama 地址：``OLLAMA_HOST`` 常写成 ``127.0.0.1:11434``（无 scheme）。

    @param raw 原始地址
    @return 带 scheme 的地址
    """
    value = raw.strip().rstrip("/")
    if not value:
        return DEFAULT_OLLAMA_BASE_URL
    if "://" not in value:
        value = f"http://{value}"
    return value


def provider_api_key(provider: str) -> tuple[str, str]:
    """
    读取该 provider 的 API Key。

    @param provider provider 名
    @return (key, 命中的环境变量名)
    """
    load_llgraph_env()
    return _env_first(_KEY_ENVS.get(provider, ()))


def provider_base_url(provider: str) -> str:
    """
    读取该 provider 的 base_url 覆盖（未配置时为空；ollama 有默认值）。

    @param provider provider 名
    @return base_url
    """
    load_llgraph_env()
    value, _name = _env_first(_BASE_URL_ENVS.get(provider, ()))
    if provider == PROVIDER_OLLAMA:
        return normalize_ollama_base_url(value)
    return value.rstrip("/")


def ollama_reachable(base_url: str = "", *, timeout: float = 0.5) -> bool:
    """
    本地 ollama 是否在听（只做 TCP 连接，不发 HTTP）。

    结果按地址缓存 ``_REACHABLE_TTL_SEC``：模型名解析是热路径（每轮多次），
    没配任何凭据时不能每次都去连一遍端口。

    @param base_url ollama 地址；空则用默认
    @param timeout 连接超时秒
    @return 是否可连
    """
    url = normalize_ollama_base_url(base_url or DEFAULT_OLLAMA_BASE_URL)
    now = time.monotonic()
    cached = _reachable_cache.get(url)
    if cached is not None and now - cached[1] < _REACHABLE_TTL_SEC:
        return cached[0]

    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 11434
    try:
        with socket.create_connection((host, port), timeout=timeout):
            ok = True
    except OSError:
        ok = False
    _reachable_cache[url] = (ok, now)
    return ok


def ollama_autodetect_enabled() -> bool:
    """
    是否允许「本机端口在听就当 ollama 可用」。

    @return 默认 True；``LLGRAPH_OLLAMA_AUTODETECT=0`` 关闭
    """
    load_llgraph_env()
    raw = os.getenv(ENV_OLLAMA_AUTODETECT, "").strip().lower()
    if not raw:
        return True
    return raw not in ("0", "false", "no", "off")


def provider_credentials_present(provider: str) -> bool:
    """
    该 provider 现在能不能用。

    @param provider provider 名
    @return 网关看 base_url+key；官方三家看 Key；ollama 看地址是否显式配置
    """
    if provider == PROVIDER_GATEWAY:
        return gateway_credentials_present()
    if provider == PROVIDER_OLLAMA:
        load_llgraph_env()
        value, _name = _env_first(_BASE_URL_ENVS[PROVIDER_OLLAMA])
        return bool(value)
    key, _name = provider_api_key(provider)
    return bool(key)


def detect_provider(model_id: str | None = None) -> tuple[str | None, str]:
    """
    只看环境变量与模型 id 选 provider（agent.json 那层在 llm_settings）。

    顺序：``LLGRAPH_PROVIDER`` > 网关凭据 > 模型 id 指向且有凭据的那家 >
    唯一配了 Key 的官方厂商 > 显式配了地址或本机在听的 ollama。

    @param model_id 当前模型 id（可为空）
    @return (provider 或 None, 选中理由)
    """
    load_llgraph_env()
    raw = os.getenv(ENV_PROVIDER, "").strip()
    if raw:
        name = normalize_provider_name(raw)
        if name is None:
            raise RuntimeError(
                f"{ENV_PROVIDER}={raw} 不认识；可选：{', '.join(KNOWN_PROVIDERS)}"
            )
        return name, f"环境变量 {ENV_PROVIDER}"

    if gateway_credentials_present():
        return PROVIDER_GATEWAY, f"网关凭据 {ENV_API_BASE_URL} + {ENV_API_KEY}"

    guess = provider_from_model_id(model_id)
    if guess is not None and provider_credentials_present(guess):
        return guess, f"模型 id {model_id}"

    for provider in (PROVIDER_ANTHROPIC, PROVIDER_OPENAI, PROVIDER_GEMINI):
        if provider_credentials_present(provider):
            _key, name = provider_api_key(provider)
            return provider, f"环境变量 {name}"

    if provider_credentials_present(PROVIDER_OLLAMA):
        return PROVIDER_OLLAMA, "环境变量 LLGRAPH_OLLAMA_BASE_URL / OLLAMA_HOST"
    if ollama_autodetect_enabled() and ollama_reachable():
        return PROVIDER_OLLAMA, f"本机 ollama（{DEFAULT_OLLAMA_BASE_URL}）"

    return None, ""


def tool_bind_kwargs(provider: str | None) -> dict[str, object]:
    """
    bind_tools 能传给这家的厂商专属参数。

    @param provider provider 名；None 视为网关（历史默认）
    @return kwargs
    """
    name = provider if provider in KNOWN_PROVIDERS else PROVIDER_GATEWAY
    if name in (PROVIDER_GATEWAY, PROVIDER_ANTHROPIC, PROVIDER_OPENAI):
        return {"parallel_tool_calls": True, "tool_choice": "auto"}
    return {}


def provider_supports_prompt_cache(provider: str | None) -> bool:
    """
    这家认不认 Anthropic 的 ``cache_control`` 断点。

    OpenAI / Gemini / Ollama 的请求体里没有这个字段：打上去要么被拒，
    要么在工具定义里多出一段没人认的 JSON。

    @param provider provider 名；None 视为网关
    @return 是否可打断点
    """
    name = provider if provider in KNOWN_PROVIDERS else PROVIDER_GATEWAY
    return name in (PROVIDER_GATEWAY, PROVIDER_ANTHROPIC)


def missing_credentials_error(model_id: str | None = None) -> RuntimeError:
    """
    一家都没配时的报错（把四条路一起说清楚，别只提网关）。

    @param model_id 当前模型 id
    @return RuntimeError
    """
    model_hint = f"当前模型 {model_id}；" if model_id else ""
    return RuntimeError(
        "未找到任何模型入口凭据。"
        f"{model_hint}四种任选一种：\n"
        f"  1. OpenAI 兼容网关：{ENV_API_BASE_URL} + {ENV_API_KEY}\n"
        f"  2. 官方 Key：{_KEY_ENV_HINT}\n"
        f"  3. 本地 Ollama：启动 ollama serve（默认 {DEFAULT_OLLAMA_BASE_URL}），"
        f"或设 LLGRAPH_OLLAMA_BASE_URL\n"
        f"  4. 显式指定：{ENV_PROVIDER}=<{'|'.join(KNOWN_PROVIDERS)}>\n"
        f"配置文件示例见 examples/llgraph.env.example"
    )


def resolve_provider_settings(
    *,
    provider: str | None = None,
    model_id: str | None = None,
) -> ProviderSettings:
    """
    解析这轮实际使用的 provider 与凭据；缺凭据时抛出可操作的错误。

    @param provider 显式 provider（agent.json / 调用方）；None 时自动探测
    @param model_id 当前模型 id
    @return ProviderSettings
    """
    source = "调用方指定"
    name = normalize_provider_name(provider) if provider else None
    if provider and name is None:
        raise RuntimeError(
            f"provider={provider} 不认识；可选：{', '.join(KNOWN_PROVIDERS)}"
        )
    if name is None:
        name, source = detect_provider(model_id)
    if name is None:
        raise missing_credentials_error(model_id)

    if name == PROVIDER_GATEWAY:
        if not gateway_credentials_present():
            # 网关也是「一家都没探到」时的落点，所以这里把四条路一起说，
            # 而不是只报缺哪两个网关变量。
            raise missing_credentials_error(model_id)
        settings = get_llgraph_settings()
        return ProviderSettings(
            provider=PROVIDER_GATEWAY,
            api_key=settings["api_key"],
            base_url=settings["base_url"],
            source=source,
        )

    if name == PROVIDER_OLLAMA:
        base_url = provider_base_url(PROVIDER_OLLAMA)
        return ProviderSettings(
            provider=PROVIDER_OLLAMA,
            api_key="",
            base_url=base_url,
            source=source,
        )

    key, key_env = provider_api_key(name)
    if not key:
        candidates = " / ".join(_KEY_ENVS.get(name, ()))
        raise RuntimeError(
            f"provider={name} 缺 API Key：请设置 {candidates}（来源：{source}）。"
            f"或改用 OpenAI 兼容网关（{ENV_API_BASE_URL} + {ENV_API_KEY}）"
            f"、本地 Ollama（{ENV_PROVIDER}=ollama）。"
        )
    return ProviderSettings(
        provider=name,
        api_key=key,
        base_url=provider_base_url(name),
        source=source if source != "调用方指定" else f"调用方指定（Key 来自 {key_env}）",
    )
