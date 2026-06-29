#!/usr/bin/env python3
"""探测各模型 tool_calls 回包形态（对比 raw LLM vs 入站 normalize vs StateGraph 路径）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage


def _model_ids(workspace: Path) -> list[str]:
    from llgraph.core.gateway_models import catalog_model_ids

    ids = catalog_model_ids(workspace)
    if ids:
        return ids
    return [
        "kimi-k2.6",
        "kimi-k2.5",
        "glm-5",
        "claude-sonnet-4-6",
        "gpt-5.4",
        "deepseek-v4-flash",
    ]


def _probe_one(
    workspace: Path,
    model_id: str,
    *,
    compare_react: bool,
) -> dict[str, object]:
    from llgraph.adapters.inbound import classify_tool_call_response, normalize_ai_response
    from llgraph.adapters.inbound.profile import resolve_inbound_profile
    from llgraph.context.message_normalize import prepare_messages_for_llm_dispatch
    from llgraph.core.agent import build_system_prompt
    from llgraph.core.llm import create_gateway_llm
    from llgraph.core.llm_settings import set_runtime_model
    from llgraph.core.tools import get_agent_tools

    set_runtime_model(model_id)
    profile = resolve_inbound_profile(workspace, model_id)
    sys_prompt = build_system_prompt(workspace, allow_write=False)
    tools = get_agent_tools(workspace_root=workspace, allow_write=False)
    tool_subset = [t for t in tools if getattr(t, "name", "") in (
        "glob_files",
        "grep_files",
        "list_directory",
    )][:3] or tools[:3]

    user = HumanMessage(
        content=(
            "工作区里 dataw 目录下谁维护工作流？"
            "请用 glob_files 查找 OWNER 或 README，不要只描述要做什么。"
        )
    )
    prepared = prepare_messages_for_llm_dispatch(
        [user],
        agent_system_content=sys_prompt,
        workspace=workspace,
        model_id=model_id,
    )

    llm = create_gateway_llm(workspace)
    bound = llm.bind_tools(
        tool_subset,
        parallel_tool_calls=True,
        tool_choice="auto",
    )
    raw = bound.invoke(prepared)

    raw_cls = classify_tool_call_response(raw)
    normalized = normalize_ai_response(raw, workspace, model_id)
    norm_cls = classify_tool_call_response(normalized)

    result: dict[str, object] = {
        "model_id": model_id,
        "inbound_profile": profile.summary(),
        "raw_structured": raw_cls["structured_tool_calls"],
        "raw_kimi_tokens": raw_cls["kimi_tokens_in_content"],
        "raw_invalid": raw_cls["has_invalid_tool_calls"],
        "norm_structured": norm_cls["structured_tool_calls"],
        "norm_kimi_tokens": norm_cls["kimi_tokens_in_content"],
        "ok_after_normalize": bool(
            norm_cls["structured_tool_calls"] and not norm_cls["kimi_tokens_in_content"]
        ),
        "error": None,
    }

    if compare_react:
        from llgraph.core.react_graph import build_react_graph
        from llgraph.context.message_normalize import make_prompt_normalizer

        graph = build_react_graph(
            llm,
            tool_subset,
            prompt=make_prompt_normalizer(sys_prompt, workspace),
            workspace=workspace,
        )
        state = graph.invoke({"messages": [user]})
        msgs = list((state or {}).get("messages") or [])
        last_ai = None
        for m in reversed(msgs):
            if getattr(m, "type", "") == "ai" or m.__class__.__name__ == "AIMessage":
                last_ai = m
                break
        if last_ai is not None:
            from langchain_core.messages import AIMessage, ToolMessage

            if isinstance(last_ai, AIMessage):
                react_cls = classify_tool_call_response(last_ai)
                result["react_structured"] = react_cls["structured_tool_calls"]
                result["react_kimi_tokens"] = react_cls["kimi_tokens_in_content"]
                result["react_tool_msgs"] = sum(
                    1 for m in msgs if isinstance(m, ToolMessage)
                )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="各模型 tool_calls 回包探测")
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--model", default=None, help="仅探测指定模型")
    parser.add_argument(
        "--compare-react",
        action="store_true",
        help="额外跑一轮 StateGraph invoke（更耗 token）",
    )
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()

    from llgraph.config.config import get_llgraph_settings, load_llgraph_env

    load_llgraph_env()
    if not get_llgraph_settings().get("api_key"):
        print("未配置 LLGRAPH_API_KEY，无法 API 探测", file=sys.stderr)
        return 1

    model_ids = [args.model] if args.model else _model_ids(workspace)
    print(
        f"tool_calls 探测  workspace={workspace}  models={len(model_ids)}"
        f"  compare_react={args.compare_react}\n"
    )
    print(
        f"{'model':<22} {'raw TC':>6} {'raw Kimi':>8} {'norm TC':>7} {'norm OK':>7}  profile"
    )
    print("-" * 88)

    all_ok = True
    for model_id in model_ids:
        try:
            row = _probe_one(workspace, model_id, compare_react=args.compare_react)
        except Exception as exc:
            all_ok = False
            print(f"{model_id:<22} {'ERR':>6} {'':>8} {'':>7} {'':>7}  {exc}")
            continue

        ok = bool(row["ok_after_normalize"]) or (
            row["raw_structured"] and not row["raw_kimi_tokens"]
        )
        if not ok:
            all_ok = False
        status = "OK" if ok else "FAIL"
        print(
            f"{model_id:<22} "
            f"{str(row['raw_structured']):>6} "
            f"{str(row['raw_kimi_tokens']):>8} "
            f"{str(row['norm_structured']):>7} "
            f"{status:>7}  "
            f"{row['inbound_profile']}"
        )
        if args.compare_react and "react_structured" in row:
            print(
                f"  └ StateGraph: structured={row.get('react_structured')} "
                f"kimi_tokens={row.get('react_kimi_tokens')} "
                f"tool_msgs={row.get('react_tool_msgs')}"
            )

    print(
        "\n说明: raw TC=网关/LangChain 原始 AIMessage.tool_calls；"
        "raw Kimi=content 含原生 token；norm OK=normalize 后可路由 tools。"
    )
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
