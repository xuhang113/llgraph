"""按模型解析上下文窗口上限（压缩预算与 /context 展示）。"""

from __future__ import annotations

import re
from pathlib import Path

from llgraph.core.gateway_models import load_model_catalog
from llgraph.core.llm_settings import resolve_effective_model

# 未在目录/启发式命中时的默认值（现代模型多为 200K）
DEFAULT_CONTEXT_WINDOW = 200_000

# 模型名启发式（顺序优先）
_CONTEXT_WINDOW_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"deepseek-v4", re.I), 1_000_000),
    (re.compile(r"claude-(opus|sonnet|haiku)-4", re.I), 200_000),
    (re.compile(r"claude-3\.7|claude-3-7", re.I), 200_000),
    (re.compile(r"gpt-5", re.I), 200_000),
    (re.compile(r"gpt-4\.1|gpt-4-1", re.I), 200_000),
    (re.compile(r"kimi-k2", re.I), 262_144),
    (re.compile(r"gemini-2\.5|gemini-2-5", re.I), 200_000),
    (re.compile(r"minimax-m2", re.I), 204_800),
    (re.compile(r"glm-5", re.I), 202_752),
    (re.compile(r"deepseek", re.I), 128_000),
    (re.compile(r"minimax", re.I), 204_800),
    (re.compile(r"kimi", re.I), 128_000),
]


def parse_context_window_value(raw: object) -> int | None:
    """
    解析 context_window 配置值。

    @param raw 配置原始值
    @return token 数或 None
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip().lower().replace("_", "").replace(",", "")
        if not text:
            return None
        mult = 1
        if text.endswith("k"):
            mult = 1000
            text = text[:-1]
        elif text.endswith("m"):
            mult = 1_000_000
            text = text[:-1]
        try:
            return max(10_000, int(float(text) * mult))
        except ValueError:
            return None
    try:
        return max(10_000, int(raw))
    except (TypeError, ValueError):
        return None


def _heuristic_context_window(model_id: str) -> int | None:
    """
    按模型名推断 context 上限。

    @param model_id 模型 id
    @return token 数或 None
    """
    for pattern, window in _CONTEXT_WINDOW_PATTERNS:
        if pattern.search(model_id):
            return window
    return None


def resolve_model_context_window(
    workspace: Path | None,
    model_id: str | None = None,
    *,
    fallback: int = DEFAULT_CONTEXT_WINDOW,
) -> tuple[int, str]:
    """
    解析模型 context 窗口上限。

    优先级：agent.json llm.models 条目的 context_window → 模型名启发式 → fallback。

    @param workspace 工作区根
    @param model_id 模型 id；None 时用当前生效模型
    @param fallback 未命中时的默认值
    @return (token 上限, 来源说明)
    """
    mid = (model_id or "").strip()
    if not mid and workspace is not None:
        mid = resolve_effective_model(workspace)

    if mid:
        catalog, _ = load_model_catalog(workspace)
        for entry in catalog:
            if entry.model_id == mid and entry.context_window is not None:
                return entry.context_window, f"catalog({mid})"

        guessed = _heuristic_context_window(mid)
        if guessed is not None:
            return guessed, f"heuristic({mid})"

    return fallback, "fallback"


def format_context_window_label(tokens: int | None) -> str:
    """
    将 context 上限格式化为简短可读标签（用于 /model 列表等）。

    @param tokens token 数
    @return 如 262K、1M；None 或无效时返回空串
    """
    if tokens is None or tokens <= 0:
        return ""
    if tokens >= 1_000_000:
        whole = tokens // 1_000_000
        if tokens % 1_000_000 == 0:
            return f"{whole}M"
        return f"{tokens / 1_000_000:.1f}M"
    if tokens >= 1000:
        whole = tokens // 1000
        if tokens % 1000 == 0:
            return f"{whole}K"
        return f"{tokens / 1000:.0f}K"
    return str(tokens)


def format_context_budget_note(
    workspace: Path,
    *,
    max_tokens: int,
    source: str,
    model_id: str,
    ratio: float,
) -> str:
    """
    压缩阈值说明行。

    @param workspace 工作区根
    @param max_tokens 有效预算
    @param source 预算来源
    @param model_id 当前模型
    @param ratio 压缩比例
    @return 单行说明
    """
    compress_at = int(max_tokens * ratio)
    if source == "config":
        return (
            f"自动压缩阈值: ~{compress_at // 1000:.1f}K"
            f"（config.max_tokens_estimate × {ratio:.0%}）"
        )
    return (
        f"自动压缩阈值: ~{compress_at // 1000:.1f}K"
        f"（模型 {model_id} 上下文 ~{max_tokens // 1000:.0f}K × {ratio:.0%}）"
    )
