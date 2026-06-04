"""通过 OpenAI 兼容网关连接 LLM API（凭据来自 LLGRAPH_*）。"""

from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_anthropic import chat_models as anthropic_chat_models

from llgraph.config.config import get_llgraph_settings
from llgraph.core.llm_settings import resolve_llm_settings

_USAGE_PATCHED = False


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


def create_gateway_llm(workspace: Path | None = None) -> ChatAnthropic:
    """
    创建指向 OpenAI 兼容网关的 Chat 模型实例。

    凭据来自 LLGRAPH_API_BASE_URL / LLGRAPH_API_KEY（见 ~/.config/llgraph/llgraph.env）。
    max_tokens 来自工作区 .llgraph/agent.json → llm.max_tokens（默认 16384）。

    @param workspace 工作区根，用于读取 agent.json
    @return ChatAnthropic 实例
    """
    from pathlib import Path as _Path

    _patch_gateway_usage_metadata()
    from llgraph.core.gateway_kimi_patch import patch_gateway_kimi_reasoning_payload

    patch_gateway_kimi_reasoning_payload()
    settings = get_llgraph_settings()
    ws = _Path(workspace).expanduser().resolve() if workspace is not None else None
    llm_cfg = resolve_llm_settings(ws)
    # 不显式传 temperature：部分网关对 temperature=0.2 会返回 400
    return ChatAnthropic(
        model=llm_cfg.model,
        api_key=settings["api_key"],
        base_url=settings["base_url"],
        max_tokens=llm_cfg.max_tokens,
    )
