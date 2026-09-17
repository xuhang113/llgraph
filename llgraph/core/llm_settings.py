"""LLM 调用参数（.llgraph/agent.json 内 llm 段 + 会话 /model）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llgraph.config.config import (
    ENV_MODEL,
    configured_model_or_none,
    resolve_configured_model,
)
from llgraph.config.edit_settings import load_agent_config
from llgraph.config.providers import (
    ENV_PROVIDER,
    PROVIDER_GATEWAY,
    default_model_for_provider,
    detect_provider,
    normalize_provider_name,
)

DEFAULT_MAX_TOKENS = 16_384
DEFAULT_REQUEST_TIMEOUT_SEC = 600.0
DEFAULT_THINKING_STREAM_TIMEOUT_SEC = 180.0

_runtime_model: str | None = None


@dataclass(frozen=True)
class LlmSettings:
    """模型生成参数。"""

    model: str
    max_tokens: int
    request_timeout_sec: float
    thinking_stream_timeout_sec: float
    provider: str = PROVIDER_GATEWAY


def set_runtime_model(model_id: str | None) -> str | None:
    """
    会话内设置运行时模型（/model）。

    @param model_id 模型名；None 表示清除覆盖、回退 env/agent.json
    @return 设置后的模型 id，清除时为 None
    """
    global _runtime_model
    if model_id is None or not str(model_id).strip():
        _runtime_model = None
        return None
    _runtime_model = str(model_id).strip()
    return _runtime_model


def get_runtime_model() -> str | None:
    """当前会话运行时模型覆盖。"""
    return _runtime_model


def resolve_explicit_model(workspace: Path | None = None) -> str | None:
    """
    用户点名的模型：会话 /model > agent.json llm.model > LLGRAPH_MODEL。

    一个都没配时返回 None——此时模型名由 provider 的默认值决定。

    @param workspace 工作区根
    @return 模型 id 或 None
    """
    if _runtime_model:
        return _runtime_model

    if workspace is not None:
        cfg = load_agent_config(workspace)
        llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
        raw = llm_cfg.get("model")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()

    return configured_model_or_none()


def resolve_effective_provider(workspace: Path | None = None) -> tuple[str, str]:
    """
    解析这轮走哪家模型入口。

    优先级：agent.json llm.provider > LLGRAPH_PROVIDER > 网关凭据 >
    模型 id 指向且有凭据的那家 > 配了官方 Key 的那家 > 本地 Ollama。
    一家都探不到时返回网关（真正建客户端时才报错，只读路径不该崩）。

    @param workspace 工作区根
    @return (provider, 来源说明)
    """
    if workspace is not None:
        cfg = load_agent_config(workspace)
        llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
        name = normalize_provider_name(llm_cfg.get("provider"))
        if name is not None:
            return name, "工作区 agent.json llm.provider"

    provider, source = detect_provider(resolve_explicit_model(workspace))
    if provider is None:
        return PROVIDER_GATEWAY, "默认（未探测到任何凭据）"
    return provider, source


def resolve_effective_model(workspace: Path | None = None) -> str:
    """
    解析实际使用的模型 id。

    优先级：会话 /model > agent.json llm.model > LLGRAPH_MODEL >
    当前 provider 的默认模型。

    @param workspace 工作区根
    @return 模型 id
    """
    explicit = resolve_explicit_model(workspace)
    if explicit:
        return explicit

    provider, _source = resolve_effective_provider(workspace)
    return default_model_for_provider(provider) or resolve_configured_model()


def resolve_llm_settings(workspace: Path | None = None) -> LlmSettings:
    """
    解析 llm 配置。

    @param workspace 工作区根；None 时仅用默认
    @return LlmSettings
    """
    max_tokens = DEFAULT_MAX_TOKENS
    request_timeout_sec = DEFAULT_REQUEST_TIMEOUT_SEC
    thinking_stream_timeout_sec = DEFAULT_THINKING_STREAM_TIMEOUT_SEC
    if workspace is not None:
        cfg = load_agent_config(workspace)
        llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
        raw = llm_cfg.get("max_tokens", DEFAULT_MAX_TOKENS)
        try:
            max_tokens = max(1024, int(raw))
        except (TypeError, ValueError):
            max_tokens = DEFAULT_MAX_TOKENS
        raw_timeout = llm_cfg.get("request_timeout_sec", DEFAULT_REQUEST_TIMEOUT_SEC)
        try:
            request_timeout_sec = max(30.0, min(3600.0, float(raw_timeout)))
        except (TypeError, ValueError):
            request_timeout_sec = DEFAULT_REQUEST_TIMEOUT_SEC
        raw_thinking_timeout = llm_cfg.get(
            "thinking_stream_timeout_sec",
            DEFAULT_THINKING_STREAM_TIMEOUT_SEC,
        )
        try:
            thinking_stream_timeout_sec = max(30.0, min(900.0, float(raw_thinking_timeout)))
        except (TypeError, ValueError):
            thinking_stream_timeout_sec = DEFAULT_THINKING_STREAM_TIMEOUT_SEC

    provider, _source = resolve_effective_provider(workspace)
    return LlmSettings(
        model=resolve_effective_model(workspace),
        max_tokens=max_tokens,
        request_timeout_sec=request_timeout_sec,
        thinking_stream_timeout_sec=thinking_stream_timeout_sec,
        provider=provider,
    )


def format_model_status(workspace: Path) -> str:
    """
    当前模型配置摘要。

    @param workspace 工作区根
    @return 多行说明
    """
    effective = resolve_effective_model(workspace)
    env_model = resolve_configured_model()
    lines = [
        f"当前模型: {effective}",
    ]
    if _runtime_model:
        lines.append(f"来源: 会话 /model（覆盖 env 默认 {env_model}）")
    else:
        cfg = load_agent_config(workspace)
        llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
        if isinstance(llm_cfg.get("model"), str) and llm_cfg.get("model", "").strip():
            lines.append(f"来源: agent.json llm.model（env 默认 {env_model}）")
        elif configured_model_or_none():
            lines.append(f"来源: 环境变量 {ENV_MODEL}")
        else:
            lines.append("来源: provider 默认模型（未配 agent.json llm.model / 环境变量）")
    provider, provider_source = resolve_effective_provider(workspace)
    lines.append(f"模型入口: {provider}（{provider_source}；切换用 {ENV_PROVIDER} 或 llm.provider）")
    lines.append("切换: /model <名>  |  /model reset 恢复默认  |  列表: /model list")
    lines.append("配置: /config（agent.json 合并规则）")
    try:
        from llgraph.context.context_settings import resolve_context_settings
        from llgraph.core.model_context_window import resolve_model_context_window

        ctx = resolve_context_settings(workspace)
        window, src = resolve_model_context_window(workspace, effective)
        lines.append(
            f"上下文预算: ~{window // 1000}K（{src}；压缩阈值 ~{int(window * ctx.auto_compress_ratio) // 1000}K）"
        )
    except Exception:
        pass
    return "\n".join(lines)


def format_model_banner_suffix(workspace: Path) -> str:
    """
    启动横幅用：模型 id + 简短来源说明。

    @param workspace 工作区根
    @return 如「claude-sonnet-4-6（工作区 agent.json）」
    """
    effective = resolve_effective_model(workspace)
    provider, _source = resolve_effective_provider(workspace)
    suffix = "" if provider == PROVIDER_GATEWAY else f"，{provider}"
    if _runtime_model:
        return f"{effective}（本会话 /model 覆盖{suffix}）"
    cfg = load_agent_config(workspace)
    llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
    if isinstance(llm_cfg.get("model"), str) and llm_cfg.get("model", "").strip():
        env_model = resolve_configured_model()
        if effective != env_model:
            return f"{effective}（工作区 agent.json llm.model；env 为 {env_model}{suffix}）"
        return f"{effective}（工作区 agent.json llm.model{suffix}）"
    if not configured_model_or_none():
        return f"{effective}（{provider} 默认模型）"
    return f"{effective}（环境变量 {ENV_MODEL}{suffix}）"
