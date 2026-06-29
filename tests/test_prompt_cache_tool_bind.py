"""Prompt Cache 与 bind_tools 顺序：cache_control 须在 bind_tools 之后。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage

from llgraph.core.llm import create_gateway_llm
from llgraph.core.llm_settings import set_runtime_model
from llgraph.core.prompt_cache import apply_prompt_cache_to_llm
from llgraph.core.react_graph import _bind_tools_if_needed
from llgraph.core.tools import get_agent_tools


def test_cache_before_bind_tools_drops_tools_from_payload() -> None:
    """apply_prompt_cache_to_llm 先于 bind_tools 时，HTTP payload 不含 tools（已知坑）。"""
    ws = Path(__file__).resolve().parents[1] / "examples" / "default-workspace"
    set_runtime_model("deepseek-v4-pro")
    llm = create_gateway_llm(ws)
    tools = get_agent_tools(workspace_root=ws, allow_write=False)[:2]
    cached = apply_prompt_cache_to_llm(llm, ws)
    bound = _bind_tools_if_needed(cached, tools)

    payloads: list[int] = []
    from langchain_anthropic import chat_models as acm

    original = acm.ChatAnthropic._get_request_payload

    def _capture(self, messages, *args, **kwargs):
        payload = original(self, messages, *args, **kwargs)
        payloads.append(len(payload.get("tools") or []))
        return payload

    acm.ChatAnthropic._get_request_payload = _capture
    try:
        bound.invoke([HumanMessage(content="hi")])
    finally:
        acm.ChatAnthropic._get_request_payload = original

    assert payloads == [0]


def test_bind_tools_then_cache_keeps_tools_in_payload() -> None:
    """build_react_graph 顺序：bind_tools 后再 apply_prompt_cache_to_llm。"""
    ws = Path(__file__).resolve().parents[1] / "examples" / "default-workspace"
    set_runtime_model("deepseek-v4-pro")
    llm = create_gateway_llm(ws)
    tools = get_agent_tools(workspace_root=ws, allow_write=False)[:2]
    bound = _bind_tools_if_needed(llm, tools)
    bound = apply_prompt_cache_to_llm(bound, ws)

    payloads: list[int] = []
    from langchain_anthropic import chat_models as acm

    original = acm.ChatAnthropic._get_request_payload

    def _capture(self, messages, *args, **kwargs):
        payload = original(self, messages, *args, **kwargs)
        payloads.append(len(payload.get("tools") or []))
        return payload

    acm.ChatAnthropic._get_request_payload = _capture
    try:
        bound.invoke([HumanMessage(content="hi")])
    finally:
        acm.ChatAnthropic._get_request_payload = original

    assert payloads == [2]
