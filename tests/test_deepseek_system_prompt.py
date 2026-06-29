"""thinking 关闭时不注入 thinking 约束 prompt。"""

from __future__ import annotations

import json
from pathlib import Path

from llgraph.core.agent import build_system_prompt


def test_build_system_prompt_omits_thinking_block_when_disabled(tmp_path: Path) -> None:
    cfg_dir = tmp_path / ".llgraph"
    cfg_dir.mkdir()
    (cfg_dir / "agent.json").write_text(
        json.dumps(
            {
                "llm": {
                    "model": "deepseek-v4-pro",
                    "thinking_defaults": {"enabled": False},
                    "models": [{"id": "deepseek-v4-pro", "thinking": False}],
                }
            }
        ),
        encoding="utf-8",
    )
    prompt = build_system_prompt(tmp_path, allow_write=False)
    assert "网关 thinking 已启用" not in prompt
    assert "thinking/reasoning 仅作内部推理" not in prompt
