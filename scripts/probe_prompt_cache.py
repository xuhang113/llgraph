#!/usr/bin/env python3
"""探测 Gateway prompt cache：两轮相同前缀请求，检查 usage 是否含 cache 字段。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _usage_summary(msg: AIMessage) -> dict:
    from llgraph.display.execution_log import _usage_dict_from_mapping

    raw = getattr(msg, "usage_metadata", None)
    if raw is None:
        meta = getattr(msg, "response_metadata", None) or {}
        if isinstance(meta, dict):
            raw = meta.get("usage_metadata") or meta.get("usage")
    part = _usage_dict_from_mapping(raw)
    reported = bool(part.pop("_cache_reported", 0))
    return {
        "normalized": part,
        "cache_reported": reported,
        "usage_metadata": raw if isinstance(raw, dict) else str(raw),
        "response_metadata_usage": (
            (getattr(msg, "response_metadata", None) or {}).get("usage")
        ),
    }


def _stable_system_block(text: str) -> SystemMessage:
    from llgraph.core.prompt_cache import build_cache_control
    from llgraph.core.prompt_cache_settings import resolve_prompt_cache_settings

    settings = resolve_prompt_cache_settings(None)
    cc = build_cache_control(settings)
    return SystemMessage(
        content=[
            {
                "type": "text",
                "text": text,
                "cache_control": cc,
            }
        ],
    )


def probe_model(workspace: Path, model_id: str) -> dict:
    from llgraph.config.config import load_llgraph_env
    from llgraph.core.llm import create_gateway_llm
    from llgraph.core.llm_settings import set_runtime_model
    from llgraph.core.prompt_cache import apply_prompt_cache_to_llm, tag_tools_for_prompt_cache
    from llgraph.core.prompt_cache_settings import (
        prompt_cache_enabled_for_model,
        resolve_prompt_cache_settings,
    )
    from llgraph.core.tools import get_agent_tools

    load_llgraph_env()
    set_runtime_model(model_id)

    settings = resolve_prompt_cache_settings(workspace)
    cache_on = prompt_cache_enabled_for_model(workspace, model_id) and settings.enabled

    llm = create_gateway_llm(workspace)
    if cache_on:
        llm = apply_prompt_cache_to_llm(llm, workspace)

    stable = (
        "你是编程助手。以下是一段用于 prompt cache 探测的稳定系统前缀。\n"
        + ("Long stable workspace policy prefix for cache probe.\n" * 80)
        + "END_STABLE_PREFIX"
    )
    sys_msg = _stable_system_block(stable)

    tools = get_agent_tools(
        workspace_root=workspace,
        allow_write=False,
        web_search_enabled=False,
    )
    if cache_on and settings.tag_tools:
        from llgraph.core.prompt_cache import build_cache_control

        tools = tag_tools_for_prompt_cache(tools, build_cache_control(settings))

    llm_with_tools = llm.bind_tools(tools) if tools else llm

    round1_msgs = [
        sys_msg,
        HumanMessage(content="第一轮：回复 OK1，不要调用工具。"),
    ]
    round2_msgs = round1_msgs + [
        AIMessage(content="OK1"),
        HumanMessage(content="第二轮：回复 OK2，不要调用工具。"),
    ]

    results: dict = {
        "model": model_id,
        "prompt_cache_config_enabled": cache_on,
        "rounds": [],
    }

    for label, msgs in (("round1", round1_msgs), ("round2", round2_msgs)):
        try:
            ai = llm_with_tools.invoke(msgs)
        except Exception as exc:
            results["rounds"].append({"label": label, "error": str(exc)})
            continue
        if not isinstance(ai, AIMessage):
            ai = AIMessage(content=str(ai))
        usage = _usage_summary(ai)
        payload_hint = {}
        try:
            payload = llm_with_tools._get_request_payload(msgs)
            sys_part = payload.get("system")
            if isinstance(sys_part, list) and sys_part:
                first = sys_part[0]
                payload_hint["system_has_cache_control"] = (
                    isinstance(first, dict) and "cache_control" in first
                )
            payload_hint["top_level_cache_control"] = "cache_control" in payload
        except Exception as exc:
            payload_hint["payload_error"] = str(exc)
        results["rounds"].append(
            {
                "label": label,
                "text_preview": (ai.content or "")[:80],
                "usage": usage,
                "payload_hint": payload_hint,
            },
        )
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="探测 prompt cache usage")
    parser.add_argument(
        "-C",
        "--workspace",
        default=".",
        help="工作区（读 agent.json；默认当前目录）",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["kimi-k2.6", "claude-sonnet-4-6", "gpt-5.4"],
        help="要探测的模型 id",
    )
    args = parser.parse_args()
    workspace = Path(args.workspace).expanduser().resolve()

    print(f"workspace: {workspace}")
    print("---")
    for mid in args.models:
        print(f"\n### model: {mid}")
        try:
            report = probe_model(workspace, mid)
        except Exception as exc:
            print(f"  FATAL: {exc}")
            continue
        print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n--- done ---")


if __name__ == "__main__":
    main()
