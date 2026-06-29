"""thinking 请求参数：支持 thinking 的模型关闭时须显式 disabled。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llgraph.core.model_thinking import (
    is_thinking_enabled,
    resolve_model_thinking_payload,
    set_runtime_thinking,
)


@pytest.fixture
def thinking_off_workspace(tmp_path: Path) -> Path:
    cfg_dir = tmp_path / ".llgraph"
    cfg_dir.mkdir()
    (cfg_dir / "agent.json").write_text(
        json.dumps(
            {
                "llm": {
                    "thinking_defaults": {"enabled": False},
                    "models": [
                        {"id": "deepseek-v4-pro", "thinking": False},
                        {"id": "kimi-k2.6", "thinking": False},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


@pytest.mark.parametrize("model_id", ["deepseek-v4-pro", "kimi-k2.6", "claude-sonnet-4-6"])
def test_thinking_off_sends_disabled_for_supported_models(
    thinking_off_workspace: Path,
    model_id: str,
) -> None:
    payload = resolve_model_thinking_payload(thinking_off_workspace, model_id)
    assert payload == {"type": "disabled"}
    assert is_thinking_enabled(thinking_off_workspace, model_id) is False


@pytest.mark.parametrize("model_id", ["deepseek-v4-flash", "kimi-k2.6", "glm-5"])
def test_runtime_off_sends_disabled(model_id: str) -> None:
    set_runtime_thinking(False)
    try:
        payload = resolve_model_thinking_payload(None, model_id)
        assert payload == {"type": "disabled"}
    finally:
        set_runtime_thinking(None)


def test_runtime_off_omits_thinking_for_unsupported_model() -> None:
    set_runtime_thinking(False)
    try:
        payload = resolve_model_thinking_payload(None, "gpt-5.4")
        assert payload is None
    finally:
        set_runtime_thinking(None)
