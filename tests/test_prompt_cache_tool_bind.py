"""Prompt Cache 与 bind_tools 顺序：cache_control 须在 bind_tools 之后。"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from llgraph.core.llm import create_gateway_llm
from llgraph.core.llm_settings import set_runtime_model
from llgraph.core.prompt_cache import apply_prompt_cache_to_llm
from llgraph.core.react_graph import _bind_tools_if_needed
from llgraph.core.tools import get_agent_tools

WORKSPACE = Path(__file__).resolve().parents[1] / "examples" / "default-workspace"


class _StopBeforeHttp(Exception):
    """payload 已构造完成，不需要真发 HTTP。"""


def _tools_in_outbound_payload(runnable: Any) -> list[int]:
    """
    记录一次 invoke 构造出的 payload 里 tools 条数，构造完即中断，不打网关。

    @param runnable 绑定好的 LLM runnable
    @return 每次 payload 构造得到的 tools 条数
    """
    from langchain_anthropic import chat_models as acm

    counts: list[int] = []
    original = acm.ChatAnthropic._get_request_payload

    def _capture(self, messages, *args, **kwargs):  # type: ignore[no-untyped-def]
        payload = original(self, messages, *args, **kwargs)
        counts.append(len(payload.get("tools") or []))
        raise _StopBeforeHttp

    acm.ChatAnthropic._get_request_payload = _capture
    try:
        with contextlib.suppress(_StopBeforeHttp):
            runnable.invoke([HumanMessage(content="hi")])
    finally:
        acm.ChatAnthropic._get_request_payload = original
    return counts


def test_cache_before_bind_tools_drops_tools_from_payload() -> None:
    """apply_prompt_cache_to_llm 先于 bind_tools 时，HTTP payload 不含 tools（已知坑）。"""
    set_runtime_model("deepseek-v4-pro")
    try:
        llm = create_gateway_llm(WORKSPACE)
        tools = get_agent_tools(workspace_root=WORKSPACE, allow_write=False)[:2]
        cached = apply_prompt_cache_to_llm(llm, WORKSPACE)
        bound = _bind_tools_if_needed(cached, tools)
        assert _tools_in_outbound_payload(bound) == [0]
    finally:
        set_runtime_model(None)


def test_bind_tools_then_cache_keeps_tools_in_payload() -> None:
    """build_react_graph 顺序：bind_tools 后再 apply_prompt_cache_to_llm。"""
    set_runtime_model("deepseek-v4-pro")
    try:
        llm = create_gateway_llm(WORKSPACE)
        tools = get_agent_tools(workspace_root=WORKSPACE, allow_write=False)[:2]
        bound = _bind_tools_if_needed(llm, tools)
        bound = apply_prompt_cache_to_llm(bound, WORKSPACE)
        assert _tools_in_outbound_payload(bound) == [2]
    finally:
        set_runtime_model(None)
