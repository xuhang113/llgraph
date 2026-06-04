"""LLM 调用参数（.llgraph/agent.json 内 llm 段 + 会话 /model）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llgraph.config.config import ENV_MODEL, get_llgraph_settings
from llgraph.config.edit_settings import load_agent_config

DEFAULT_MAX_TOKENS = 16_384

_runtime_model: str | None = None


@dataclass(frozen=True)
class LlmSettings:
    """Gateway 模型生成参数。"""

    model: str
    max_tokens: int


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


def resolve_effective_model(workspace: Path | None = None) -> str:
    """
    解析实际使用的模型 id。

    优先级：会话 /model > agent.json llm.model > LLGRAPH_MODEL。

    @param workspace 工作区根
    @return 模型 id
    """
    if _runtime_model:
        return _runtime_model

    if workspace is not None:
        cfg = load_agent_config(workspace)
        llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
        raw = llm_cfg.get("model")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()

    settings = get_llgraph_settings()
    return settings["model"]


def resolve_llm_settings(workspace: Path | None = None) -> LlmSettings:
    """
    解析 llm 配置。

    @param workspace 工作区根；None 时仅用默认
    @return LlmSettings
    """
    max_tokens = DEFAULT_MAX_TOKENS
    if workspace is not None:
        cfg = load_agent_config(workspace)
        llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
        raw = llm_cfg.get("max_tokens", DEFAULT_MAX_TOKENS)
        try:
            max_tokens = max(1024, int(raw))
        except (TypeError, ValueError):
            max_tokens = DEFAULT_MAX_TOKENS

    return LlmSettings(
        model=resolve_effective_model(workspace),
        max_tokens=max_tokens,
    )


def format_model_status(workspace: Path) -> str:
    """
    当前模型配置摘要。

    @param workspace 工作区根
    @return 多行说明
    """
    effective = resolve_effective_model(workspace)
    env_model = get_llgraph_settings()["model"]
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
        else:
            lines.append(f"来源: 环境变量 {ENV_MODEL}")
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
    if _runtime_model:
        return f"{effective}（本会话 /model 覆盖）"
    cfg = load_agent_config(workspace)
    llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
    if isinstance(llm_cfg.get("model"), str) and llm_cfg.get("model", "").strip():
        env_model = get_llgraph_settings()["model"]
        if effective != env_model:
            return f"{effective}（工作区 agent.json llm.model；env 为 {env_model}）"
        return f"{effective}（工作区 agent.json llm.model）"
    return f"{effective}（环境变量 {ENV_MODEL}）"
