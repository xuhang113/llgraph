"""包内 prompt YAML 加载与组装。"""

from pathlib import Path

from llgraph.loaders.prompt_loader import (
    compose_agent_system_prompt,
    compose_plan_planner_role,
    compose_plan_synthesize_system,
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
    assert (root / "plan" / "planner.yaml").is_file()
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
        survey_interactive_enabled=False,
    )
    assert "test-model" in text
    assert hint in text
    assert "工具选型" in hint
    assert "grep_files" in text
    assert "ReAct" in text
    assert "Cursor" in text


def test_compose_search_order_hint_includes_routing_when_indexed() -> None:
    _, hint = compose_search_order_hint(index_ready=True)
    assert "工具选型" in hint
    assert "常见多余 step" in hint
    assert "search_code_parallel" in hint


def test_compose_plan_and_thought_helpers() -> None:
    planner = compose_plan_planner_role(workspace=Path("/tmp/ws"))
    assert "Planner" in planner
    assert "/tmp/ws" in planner

    synth = compose_plan_synthesize_system()
    assert synth.strip()

    header = compose_thought_block_header(emit_plan_line=True)
    assert "【规划】" in header

    builtin = compose_thought_builtin_retrieval()
    assert "grep_files" in builtin
