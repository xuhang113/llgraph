"""包内 prompt YAML 加载与组装。"""

from pathlib import Path

from llgraph.loaders.prompt_loader import (
    compose_agent_system_prompt,
    compose_search_order_hint,
    compose_thought_block_header,
    compose_thought_builtin_retrieval,
    load_prompt_module,
    prompts_root,
)


def test_prompts_root_contains_yaml_modules() -> None:
    root = prompts_root()
    assert (root / "agent" / "identity.yaml").is_file()
    assert (root / "agent" / "workflow.yaml").is_file()
    assert (root / "thought" / "block.yaml").is_file()


def test_load_prompt_module_returns_dict() -> None:
    data = load_prompt_module("agent", "identity")
    assert isinstance(data, dict)
    assert "intro" in data
    assert data["intro"].strip()


def test_compose_agent_system_prompt_non_empty() -> None:
    tools_read, hint = compose_search_order_hint(index_ready=False)
    text = compose_agent_system_prompt(
        workspace_root=Path("/tmp/ws"),
        model_id="test-model",
        mode="只读",
        tools_read=tools_read,
        tools_write="",
        edit_hint="",
        search_order_hint=hint,
        thinking_payload=None,
        web_search_enabled=False,
        allow_write=False,
    )
    assert "test-model" in text
    assert hint in text
    assert "grep_files" in text
    assert "todo_write" in text
    assert "llgraph" in text
    assert "# Doing tasks" in text
    assert "# Using your tools" in text or "# 使用工具" in text
    assert "# Output efficiency" in text or "# 输出效率" in text
    assert "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__" in text
    # 边界前为静态原则，检索路由在动态侧
    static, _, dynamic = text.partition("__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__")
    assert "# Doing tasks" in static
    assert "Code search routing" in dynamic or "search_code_parallel" in dynamic
    assert len(text) < 14_000


def test_compose_search_order_hint_includes_routing_when_indexed() -> None:
    _, hint = compose_search_order_hint(index_ready=True)
    assert "Code search routing" in hint or "search_code_parallel" in hint
    assert "Avoid" in hint or "search_code_parallel" in hint


def test_compose_thought_helpers() -> None:
    header = compose_thought_block_header(emit_plan_line=True)
    assert "【规划】" in header

    builtin = compose_thought_builtin_retrieval()
    assert "grep" in builtin.lower() or "tool" in builtin.lower()
