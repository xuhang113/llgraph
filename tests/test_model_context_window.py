"""模型 context 窗口启发式与 catalog 解析。"""

from __future__ import annotations

import pytest

from llgraph.core.model_context_window import (
    format_context_window_label,
    parse_context_window_value,
    resolve_model_context_window,
)


@pytest.mark.parametrize(
    ("model_id", "expected_tokens", "label"),
    [
        ("deepseek-v4-flash", 1_000_000, "1M"),
        ("DeepSeek-V4", 1_000_000, "1M"),
        ("deepseek-chat", 128_000, "128K"),
        ("kimi-k2.6", 262_144, "262K"),
        ("claude-sonnet-4-6", 200_000, "200K"),
        ("glm-5", 202_752, "203K"),
        ("unknown-model-xyz", 200_000, "200K"),
    ],
)
def test_resolve_context_window(model_id: str, expected_tokens: int, label: str) -> None:
    tokens, source = resolve_model_context_window(None, model_id, fallback=200_000)
    assert tokens == expected_tokens
    if model_id == "unknown-model-xyz":
        assert source == "fallback"
    else:
        assert source.startswith("heuristic")
    assert format_context_window_label(tokens) == label


def test_parse_context_window_m_and_k() -> None:
    assert parse_context_window_value("1M") == 1_000_000
    assert parse_context_window_value("256K") == 256_000
    assert parse_context_window_value(1_000_000) == 1_000_000


def test_deepseek_v4_not_256k() -> None:
    tokens, _ = resolve_model_context_window(None, "deepseek-v4-flash")
    assert tokens == 1_000_000
    assert tokens != 262_144
    assert tokens != 256_000
