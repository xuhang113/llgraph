"""跨模型 thinking 落盘 / effort 拆分 / 签名块回灌。"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from llgraph.context.chat_history_repair import rehydrate_native_thinking_block
from llgraph.context.message_canonical import persist_ai_thinking_in_message
from llgraph.core.llm_response import llm_thinking_text
from llgraph.core.model_thinking import (
    split_thinking_payload,
    set_runtime_thinking,
)


def test_adaptive_splits_default_effort_high() -> None:
    body, effort = split_thinking_payload({"type": "adaptive"})
    assert body == {"type": "adaptive"}
    assert effort == "high"


def test_adaptive_respects_explicit_effort() -> None:
    body, effort = split_thinking_payload({"type": "adaptive", "effort": "max"})
    assert body == {"type": "adaptive"}
    assert effort == "max"


def test_enabled_thinking_has_no_effort() -> None:
    body, effort = split_thinking_payload({"type": "enabled", "keep": "all"})
    assert body == {"type": "enabled", "keep": "all"}
    assert effort is None


def test_persist_signature_only_thinking_block() -> None:
    """东买/Bedrock Opus：thinking 明文为空但有 signature，须落盘 thinking_blocks。"""
    msg = AIMessage(
        content=[
            {"type": "thinking", "thinking": None, "signature": "sig-abc"},
            {"type": "text", "text": "最终答案"},
        ]
    )
    fixed, changed = persist_ai_thinking_in_message(msg)
    assert changed is True
    assert fixed.content == "最终答案"
    meta = (fixed.additional_kwargs or {}).get("llgraph") or {}
    assert meta.get("thinking_redacted") is True
    assert meta.get("thinking_blocks") == [
        {"type": "thinking", "thinking": "", "signature": "sig-abc"}
    ]
    assert not meta.get("thinking_text")
    assert "签名块" in llm_thinking_text(fixed)


def test_persist_plaintext_thinking_block() -> None:
    msg = AIMessage(
        content=[
            {"type": "thinking", "thinking": "先分析问题", "signature": "sig-1"},
            {"type": "text", "text": "结论"},
        ]
    )
    fixed, _ = persist_ai_thinking_in_message(msg)
    meta = (fixed.additional_kwargs or {}).get("llgraph") or {}
    assert meta.get("thinking_text") == "先分析问题"
    assert meta.get("thinking_blocks")[0]["thinking"] == "先分析问题"
    assert meta.get("thinking_redacted") is None


def test_rehydrate_prefers_signature_blocks() -> None:
    msg = AIMessage(
        content="可见正文",
        additional_kwargs={
            "llgraph": {
                "thinking_blocks": [
                    {"type": "thinking", "thinking": "", "signature": "sig-xyz"}
                ],
                "thinking_redacted": True,
            }
        },
    )
    fixed, changed = rehydrate_native_thinking_block(msg)
    assert changed is True
    assert isinstance(fixed.content, list)
    assert fixed.content[0]["type"] == "thinking"
    assert fixed.content[0]["signature"] == "sig-xyz"
    assert fixed.content[1]["text"] == "可见正文"


def test_runtime_reset() -> None:
    set_runtime_thinking(None)
