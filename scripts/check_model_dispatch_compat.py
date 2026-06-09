#!/usr/bin/env python3
"""校验长会话在各模型 dispatch profile 下的出站消息链（可选探测 API）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage, messages_from_dict


def _load_raw_messages(path: Path) -> list[BaseMessage]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return list(messages_from_dict(rows))


def _strict_tool_violations(messages: list[BaseMessage]) -> int:
    from llgraph.context.chat_history_repair import ai_message_has_tool_calls

    count = 0
    for idx, msg in enumerate(messages):
        from langchain_core.messages import ToolMessage

        if not isinstance(msg, ToolMessage):
            continue
        if idx == 0:
            count += 1
            continue
        prev = messages[idx - 1]
        if not isinstance(prev, AIMessage) or not ai_message_has_tool_calls(prev):
            count += 1
    return count


def _missing_reasoning_in_http_payload(
    prepared: list[BaseMessage],
    workspace: Path,
    model_id: str,
) -> list[int]:
    from llgraph.core.llm import create_gateway_llm
    from llgraph.core.gateway_kimi_patch import (
        is_kimi_thinking_model,
        missing_reasoning_on_formatted_tool_assistants,
    )
    from llgraph.core.llm_settings import set_runtime_model

    set_runtime_model(model_id)
    llm = create_gateway_llm(workspace)
    payload = llm._get_request_payload(prepared)
    formatted = payload.get("messages")
    if not isinstance(formatted, list):
        return [-1]
    return missing_reasoning_on_formatted_tool_assistants(
        formatted,
        require_thinking_block=is_kimi_thinking_model(model_id),
    )


def _simulate_parallel_tool_roundtrip(
    workspace: Path,
    model_id: str,
) -> list[int]:
    """模拟并行 tool 后第二次 invoke 的 HTTP payload reasoning 缺口。"""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from llgraph.core.agent import build_system_prompt
    from llgraph.context.message_normalize import prepare_messages_for_llm_dispatch

    sys_prompt = build_system_prompt(workspace, allow_write=False)
    human = HumanMessage(content="probe parallel tools")
    ai = AIMessage(
        content="【规划】并行探测",
        tool_calls=[
            {
                "name": "grep_files",
                "args": {},
                "id": "call_1",
                "type": "tool_call",
            },
            {
                "name": "search_workspace",
                "args": {},
                "id": "call_2",
                "type": "tool_call",
            },
        ],
        additional_kwargs={"llgraph": {"thinking_text": "内部思考"}},
    )
    t1 = ToolMessage(content="a", tool_call_id="call_1", name="grep_files")
    t2 = ToolMessage(content="b", tool_call_id="call_2", name="search_workspace")
    prepared = prepare_messages_for_llm_dispatch(
        [human, ai, t1, t2],
        agent_system_content=sys_prompt,
        workspace=workspace,
        model_id=model_id,
    )
    return _missing_reasoning_in_http_payload(prepared, workspace, model_id)


def _missing_reasoning_on_tool_ai(messages: list[BaseMessage]) -> int:
    from llgraph.context.chat_history_repair import ai_message_has_tool_calls

    missing = 0
    for msg in messages:
        if not isinstance(msg, AIMessage) or not ai_message_has_tool_calls(msg):
            continue
        extra = getattr(msg, "additional_kwargs", None) or {}
        raw = extra.get("reasoning_content")
        if not isinstance(raw, str) or not raw.strip():
            missing += 1
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="多模型 dispatch 兼容性检查")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--messages",
        type=Path,
        default=None,
        help="messages.jsonl；默认 ~/.llgraph/context/<ws>/sessions/<thread>/messages.jsonl",
    )
    parser.add_argument("--thread-id", default="cli-dc2eda6d")
    parser.add_argument(
        "--probe-api",
        action="store_true",
        help="对每模型发一条极短探测（耗 token，需 LLGRAPH_API_KEY）",
    )
    parser.add_argument(
        "--probe-roundtrip",
        action="store_true",
        help="模拟并行 tool 返回后的二次 invoke，并探测各模型 API（更贴近 Kimi 400 场景）",
    )
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()

    if args.messages is not None:
        msg_path = args.messages.expanduser().resolve()
    else:
        from llgraph.session.user_storage import session_messages_path

        msg_path = session_messages_path(workspace, args.thread_id)

    if not msg_path.is_file():
        print(f"消息文件不存在: {msg_path}", file=sys.stderr)
        return 1

    raw = _load_raw_messages(msg_path)
    print(f"会话: {args.thread_id}  原始消息: {len(raw)} 条  路径: {msg_path}")

    from llgraph.core.gateway_models import catalog_model_ids
    from llgraph.context.message_dispatch_profile import (
        canonical_persist_profile,
        resolve_dispatch_profile,
    )
    from llgraph.context.chat_history_repair import (
        rebuild_provider_safe_messages,
        sanitize_chat_history,
    )

    persist = canonical_persist_profile()
    canonical, persist_report = sanitize_chat_history(raw, persist)
    print(
        f"落盘 canonical: {len(canonical)} 条"
        f"（变更: {persist_report.changed}）"
    )

    model_ids = catalog_model_ids(workspace)
    if not model_ids:
        model_ids = [
            "kimi-k2.6",
            "kimi-k2.5",
            "glm-5",
            "claude-sonnet-4-6",
            "gpt-5.4",
            "deepseek-v4-flash",
        ]

    print("\n模型出站 profile 校验（基于 canonical 历史，不污染落盘形态）:")
    print(f"{'model':<22} {'msgs':>5} {'strict':>6} {'reason':>6}  profile")
    print("-" * 72)

    all_ok = True
    for model_id in model_ids:
        profile = resolve_dispatch_profile(workspace, model_id)
        dispatched, _report = rebuild_provider_safe_messages(canonical, profile)
        strict_bad = _strict_tool_violations(dispatched)
        reason_bad = 0
        if profile.patch_tool_ai_reasoning:
            reason_bad = _missing_reasoning_on_tool_ai(dispatched)
        ok = strict_bad == 0 and reason_bad == 0
        if not ok:
            all_ok = False
        status = "OK" if ok else "FAIL"
        print(
            f"{model_id:<22} {len(dispatched):>5} {strict_bad:>6} {reason_bad:>6}  "
            f"{profile.summary()} [{status}]"
        )

    print("\n并行 tool 二次 invoke（HTTP payload reasoning_content）:")
    for model_id in model_ids:
        profile = resolve_dispatch_profile(workspace, model_id)
        if not profile.patch_tool_ai_reasoning:
            print(f"  {model_id}: SKIP（未启用 patch_reasoning_content）")
            continue
        missing = _simulate_parallel_tool_roundtrip(workspace, model_id)
        ok = not missing
        if not ok:
            all_ok = False
        status = "OK" if ok else f"FAIL missing@{missing}"
        print(f"  {model_id}: {status}")

    if args.probe_roundtrip:
        from llgraph.config.config import get_llgraph_settings, load_llgraph_env
        from llgraph.core.llm import create_gateway_llm
        from llgraph.core.llm_settings import set_runtime_model
        from llgraph.context.message_normalize import prepare_messages_for_llm_dispatch
        from llgraph.core.agent import build_system_prompt
        from llgraph.core.tools import get_agent_tools
        from llgraph.core.prompt_cache import apply_prompt_cache_to_llm
        from llgraph.core.prompt_cache_settings import (
            prompt_cache_enabled_for_model,
            resolve_prompt_cache_settings,
        )
        from llgraph.core.gateway_kimi_patch import (
            is_kimi_thinking_model,
            missing_reasoning_on_formatted_tool_assistants,
        )

        load_llgraph_env()
        settings = get_llgraph_settings()
        if not settings.get("api_key"):
            print("\n跳过 roundtrip API 探测: 未配置 LLGRAPH_API_KEY", file=sys.stderr)
        else:
            print("\n并行 tool 后二次 invoke（真实 API，各模型 1 次）:")
            sys_prompt = build_system_prompt(workspace, allow_write=False)
            cache_settings = resolve_prompt_cache_settings(workspace)
            tools = get_agent_tools(workspace_root=workspace, allow_write=False)[:5]
            for model_id in model_ids:
                set_runtime_model(model_id)
                human = HumanMessage(content="probe roundtrip")
                ai = AIMessage(
                    content="【规划】并行探测",
                    tool_calls=[
                        {
                            "name": "grep_files",
                            "args": {},
                            "id": "rt1",
                            "type": "tool_call",
                        },
                        {
                            "name": "search_files",
                            "args": {},
                            "id": "rt2",
                            "type": "tool_call",
                        },
                    ],
                    additional_kwargs={
                        "llgraph": {"thinking_text": "roundtrip probe thinking"},
                    },
                )
                t1 = ToolMessage(
                    content="tool-a",
                    tool_call_id="rt1",
                    name="grep_files",
                )
                t2 = ToolMessage(
                    content="tool-b",
                    tool_call_id="rt2",
                    name="search_files",
                )
                probe = HumanMessage(content="只回复一个词：ok")
                prepared = prepare_messages_for_llm_dispatch(
                    [human, ai, t1, t2, probe],
                    agent_system_content=sys_prompt,
                    workspace=workspace,
                    model_id=model_id,
                )
                llm = create_gateway_llm(workspace)
                if (
                    prompt_cache_enabled_for_model(workspace, model_id)
                    and cache_settings.enabled
                ):
                    llm = apply_prompt_cache_to_llm(llm, workspace)
                bound = llm.bind_tools(tools)
                profile = resolve_dispatch_profile(workspace, model_id)
                if profile.patch_tool_ai_reasoning:
                    payload = bound._get_request_payload(prepared)
                    miss = missing_reasoning_on_formatted_tool_assistants(
                        payload.get("messages") or [],
                        require_thinking_block=is_kimi_thinking_model(model_id),
                    )
                    if miss:
                        all_ok = False
                        print(f"  {model_id}: PAYLOAD FAIL missing@{miss}")
                        continue
                try:
                    out = bound.invoke(prepared)
                    text = str(getattr(out, "content", "") or "")[:60]
                    preview = text.replace("\n", " ")
                    print(f"  {model_id}: OK — {preview!r}")
                except Exception as exc:
                    all_ok = False
                    print(f"  {model_id}: FAIL — {exc}")

    if args.probe_api:
        from llgraph.config.config import get_llgraph_settings, load_llgraph_env

        load_llgraph_env()
        from llgraph.core.agent import build_system_prompt
        from llgraph.core.llm import create_gateway_llm
        from llgraph.core.llm_settings import set_runtime_model
        from llgraph.core.prompt_cache import apply_prompt_cache_to_llm
        from llgraph.core.prompt_cache_settings import (
            prompt_cache_enabled_for_model,
            resolve_prompt_cache_settings,
        )
        from llgraph.context.context_dispatch_window import trim_messages_for_dispatch_window
        from llgraph.context.context_settings import resolve_context_settings
        from llgraph.context.message_normalize import prepare_messages_for_llm_dispatch

        settings = get_llgraph_settings()
        if not settings.get("api_key"):
            print("\n跳过 API 探测: 未配置 LLGRAPH_API_KEY", file=sys.stderr)
        else:
            print("\nAPI 探测（dispatch 窗口 + 探测句，各模型 1 次）:")
            probe = HumanMessage(content="请只回复一个词：ok")
            ctx_settings = resolve_context_settings(workspace)
            sys_prompt = build_system_prompt(workspace, allow_write=False)
            cache_settings = resolve_prompt_cache_settings(workspace)
            for model_id in model_ids:
                set_runtime_model(model_id)
                profile = resolve_dispatch_profile(workspace, model_id)
                dispatched, _ = rebuild_provider_safe_messages(canonical, profile)
                if ctx_settings.dispatch_keep_user_turns > 0:
                    dispatched = trim_messages_for_dispatch_window(
                        dispatched,
                        keep_user_turns=ctx_settings.dispatch_keep_user_turns,
                    )
                batch = [*dispatched, probe]
                prepared = prepare_messages_for_llm_dispatch(
                    batch,
                    agent_system_content=sys_prompt,
                    workspace=workspace,
                    model_id=model_id,
                )
                llm = create_gateway_llm(workspace)
                if (
                    prompt_cache_enabled_for_model(workspace, model_id)
                    and cache_settings.enabled
                ):
                    llm = apply_prompt_cache_to_llm(llm, workspace)
                try:
                    out = llm.invoke(prepared)
                    text = getattr(out, "content", "") or ""
                    preview = str(text)[:80].replace("\n", " ")
                    print(f"  {model_id}: OK — {preview!r}")
                except Exception as exc:
                    all_ok = False
                    print(f"  {model_id}: FAIL — {exc}")

    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
