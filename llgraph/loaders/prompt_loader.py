"""包内内置 prompt（YAML）加载与模板渲染；不支持工作区/用户覆盖。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from llgraph.core.prompt_boundary import SYSTEM_PROMPT_DYNAMIC_BOUNDARY

_PROMPTS_ROOT = Path(__file__).resolve().parent.parent / "prompts"


class _SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def prompts_root() -> Path:
    """内置 prompt 目录（随 llgraph 包发布）。"""
    return _PROMPTS_ROOT


@lru_cache(maxsize=64)
def _read_yaml_file(path: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        return {}
    raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def load_prompt_module(module: str, name: str) -> dict[str, Any]:
    """
    读取 prompts/<module>/<name>.yaml。

    @param module 如 agent、plan、thought
    @param name 文件名（不含 .yaml）
    @return 顶层 dict；缺失文件返回 {}
    """
    path = _PROMPTS_ROOT / module / f"{name}.yaml"
    return _read_yaml_file(str(path.resolve()))


def prompt_text(module: str, name: str, key: str, *, default: str = "") -> str:
    """取 YAML 中某个 string 字段。"""
    data = load_prompt_module(module, name)
    value = data.get(key, default)
    return str(value).strip() if value is not None else default


def render_prompt(template: str, **variables: Any) -> str:
    """安全格式化 `{var}` 占位符；未知键保留原样。"""
    if not template:
        return ""
    return template.format_map(_SafeFormatDict(**variables)).strip()


def compose_agent_system_prompt(
    *,
    workspace_root: Path,
    model_id: str,
    mode: str,
    tools_read: str,
    tools_write: str,
    edit_hint: str,
    search_order_hint: str,
    thinking_payload: Any | None,
    web_search_enabled: bool,
    allow_write: bool,
) -> str:
    """
    组装 Agent 主 system prompt。

    静态前缀：Intro → System → Doing tasks → Actions → Using tools → Tone/Efficiency；
    其后经动静边界接 Session / Environment / 检索路由 / 记忆等动态段。

    @return 完整 system 文本（含动静边界标记）
    """
    identity = load_prompt_module("agent", "identity")
    display = load_prompt_module("agent", "display")
    context = load_prompt_module("agent", "context")
    shell = load_prompt_module("agent", "shell")
    tools = load_prompt_module("agent", "tools")
    workflow = load_prompt_module("agent", "workflow")

    vars_base = {
        "model_id": model_id,
        "workspace_root": workspace_root,
        "mode": mode,
        "tools_read": tools_read,
        "tools_write": tools_write,
        "edit_hint": edit_hint,
        "search_order_hint": search_order_hint,
    }

    # 静态可缓存前缀
    static_parts: list[str] = [
        render_prompt(str(identity.get("intro", "")), **vars_base),
        render_prompt(str(identity.get("model_identity_rule", "")), **vars_base),
        render_prompt(str(display.get("terminal_display", "")), **vars_base),
        render_prompt(str(workflow.get("react_loop", "")), **vars_base),
        render_prompt(str(workflow.get("doing_tasks", "")), **vars_base),
        render_prompt(str(workflow.get("investigation_depth", "")), **vars_base),
        render_prompt(str(workflow.get("investigation", "")), **vars_base),
        render_prompt(str(workflow.get("code_quality", "")), **vars_base),
        render_prompt(str(workflow.get("actions", "")), **vars_base),
        render_prompt(str(tools.get("tool_selection_summary", "")), **vars_base),
        render_prompt(str(tools.get("edit_hint", "")), **vars_base) if edit_hint else "",
        render_prompt(str(workflow.get("batch_tool_calls", "")), **vars_base),
        render_prompt(str(tools.get("parallel_invoke", "")), **vars_base),
        render_prompt(str(shell.get("run_shell", "")), **vars_base),
        render_prompt(str(identity.get("response_style", "")), **vars_base),
    ]

    if thinking_payload is not None:
        thinking = load_prompt_module("agent", "thinking")
        static_parts.append(
            render_prompt(
                str(thinking.get("constraints", "")),
                thinking_payload=thinking_payload,
                **vars_base,
            )
        )

    static = "\n\n".join(p for p in static_parts if p and p.strip())

    # 动态段（会话 / 环境 / 检索）
    dynamic_parts: list[str] = [
        render_prompt(str(context.get("manifest_and_skills", "")), **vars_base),
        render_prompt(str(context.get("session_history", "")), **vars_base),
        render_prompt(str(context.get("business_docs", "")), **vars_base),
        render_prompt(str(context.get("long_term_memory", "")), **vars_base),
        render_prompt(str(identity.get("workspace", "")), **vars_base),
        search_order_hint,
        render_prompt(str(shell.get("paths_and_spill", "")), **vars_base),
    ]
    if web_search_enabled:
        dynamic_parts.append(render_prompt(str(tools.get("web_search_hint", "")), **vars_base))

    dynamic = "\n\n".join(p for p in dynamic_parts if p and p.strip())

    if dynamic:
        return f"{static}\n\n{SYSTEM_PROMPT_DYNAMIC_BOUNDARY}\n\n{dynamic}"
    return static


def compose_search_order_hint(*, index_ready: bool) -> tuple[str, str]:
    """
    检索顺序与 tools_read 列表。

    @return (tools_read, search_order_hint)
    """
    search = load_prompt_module("agent", "search")
    tools = load_prompt_module("agent", "tools")

    chunks = [
        str(search.get("tool_routing", "")),
        str(search.get("anti_patterns", "")),
        str(search.get("path_scope", "")),
        str(search.get("tool_invoke_params", "")),
        str(search.get("code_search", "")),
        str(search.get("batch_read", "")),
        str(search.get("batch_search", "")),
    ]

    if index_ready:
        tools_read = str(tools.get("tools_read_indexed", ""))
        body = str(search.get("search_index_ready", ""))
        tail = str(search.get("glob_grep_miss_tail", ""))
        hint = "\n\n".join(p.strip() for p in [body, *chunks, tail] if p and p.strip())
    else:
        tools_read = str(tools.get("tools_read_no_index", ""))
        body = str(search.get("search_no_index", ""))
        hint = "\n\n".join(
            p.strip()
            for p in [
                body,
                str(search.get("tool_routing", "")),
                str(search.get("anti_patterns", "")),
                str(search.get("path_scope", "")),
                str(search.get("tool_invoke_params", "")),
                str(search.get("batch_read", "")),
                str(search.get("batch_search", "")),
            ]
            if p and p.strip()
        )

    return tools_read.strip(), hint.strip()


def compose_thought_block_header(*, emit_plan_line: bool) -> str:
    """Thought 注入块标题与 emit_plan_line 硬性要求。"""
    header = load_prompt_module("thought", "block")
    parts: list[str] = [str(header.get("title", "")).strip()]
    if emit_plan_line:
        line = str(header.get("emit_plan_line_requirement", "")).strip()
        if line:
            parts.extend(["", line])
    return "\n".join(p for p in parts if p)


def compose_thought_builtin_retrieval() -> str:
    """包内默认 Thought 检索规范（无工作区 .llgraph/thought 时 fallback）。"""
    return prompt_text("thought", "builtin_retrieval", "body")
