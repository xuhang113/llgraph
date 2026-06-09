"""会话 Banner 文本（终端启动面板）。"""

from __future__ import annotations

from pathlib import Path

from llgraph.loaders.rules_loader import discover_rules
from llgraph.loaders.skills_loader import discover_skills
from llgraph.loaders.thought_loader import thought_summary
from llgraph.display.trace_display import TRACE_MODE_LABELS, TraceSession

_BANNER_COMMANDS = (
    "/help · /config · /sessions · /paste · /trace · /compress · "
    "/tools · /rule · /skill · /index · exit"
)


def _resolve_banner_metrics(
    workspace: Path,
    trace_session: TraceSession,
    *,
    watch_active: bool,
    web_search_enabled: bool,
    mcp_summary: str,
) -> dict[str, str]:
    """
    汇总终端 Banner 各指标文案。

    @param workspace 工作区
    @param trace_session 追踪配置
    @param watch_active 监听是否运行
    @param web_search_enabled 联网搜索
    @param mcp_summary MCP 摘要
    @return 字段键值
    """
    rule_n = len(discover_rules(workspace))
    skill_n = len(discover_skills(workspace))
    thought_hint = thought_summary(workspace)
    try:
        from llgraph.code_index.index_settings import resolve_index_settings
        from llgraph.code_index.store import get_index_status
        from llgraph.config.logging_settings import level_name, resolve_log_level
        from llgraph.core.llm_settings import format_model_banner_suffix

        idx = get_index_status(workspace)
        idx_value = (
            f"{idx.chunk_count} chunks" if idx.exists else "未索引（llgraph index -C .）"
        )
        log_value = level_name(resolve_log_level(workspace))
        model_value = format_model_banner_suffix(workspace)
        debounce = resolve_index_settings(workspace).watch_debounce_sec
    except RuntimeError:
        idx_value = "未安装 index extra（pip install -e '.[index]'）"
        log_value = "—"
        model_value = "—"
        debounce = 5.0

    if watch_active:
        watch_value = f"已启动（保存文件后 debounce 增量索引，{debounce}s）"
    else:
        watch_value = "未启动（/watch on 或去掉 --no-watch-index）"

    web_line = ""
    if web_search_enabled:
        web_line = "已启用（/web off 可关）"
    else:
        web_line = "未启用（/web on）"

    mcp_line = mcp_summary.strip().splitlines()[0] if mcp_summary.strip() else (
        "MCP: 未加载（.llgraph/mcp.json）"
    )
    if mcp_line.startswith("MCP:"):
        mcp_value = mcp_line.replace("MCP:", "", 1).strip() or "未加载"
    else:
        mcp_value = mcp_line

    return {
        "idx": idx_value,
        "log": log_value,
        "model": model_value,
        "watch": watch_value,
        "web": web_line,
        "thought": thought_hint,
        "trace": TRACE_MODE_LABELS[trace_session.mode],
        "rules": f"{rule_n} 条",
        "skills": f"{skill_n} 个",
        "mcp": mcp_value,
    }


def print_terminal_session_banner(
    *,
    workspace: Path,
    allow_write: bool,
    thread_id: str,
    trace_session: TraceSession,
    watch_active: bool = False,
    web_search_enabled: bool = False,
    mcp_summary: str = "",
    resume_hint: str = "",
    memory_kind: str = "",
) -> None:
    """
    打印经典终端启动面板（分组 + 树形指标，对齐旧版 terminal 体验）。

    @param workspace 工作区
    @param allow_write 是否可写
    @param thread_id 线程 ID
    @param trace_session 追踪配置
    @param watch_active 索引监听
    @param web_search_enabled 联网搜索
    @param mcp_summary MCP 摘要
    @param resume_hint 恢复提示
    @param memory_kind 记忆后端说明
    """
    from llgraph.display.terminal_style import print_section, print_section_rows
    from llgraph.terminal.style import indent_line, sty
    from llgraph.session.user_storage import format_storage_location_hint

    metrics = _resolve_banner_metrics(
        workspace,
        trace_session,
        watch_active=watch_active,
        web_search_enabled=web_search_enabled,
        mcp_summary=mcp_summary,
    )
    ws_display = str(workspace.expanduser().resolve())
    file_mode = "可读写 (-w)" if allow_write else "只读（默认，不可写文件）"

    print(sty("llgraph 交互会话", "brand"), flush=True)
    print(sty(indent_line(1) + f"（{_BANNER_COMMANDS}）", "hint"), flush=True)
    print("", flush=True)

    print_section("工作区")
    workspace_rows: list[tuple[str, str, str]] = [
        ("workspace", ws_display, ""),
        ("文件", file_mode, ""),
        ("thread", thread_id, ""),
    ]
    if resume_hint.strip():
        workspace_rows.append(("恢复", resume_hint.strip(), ""))
    print_section_rows(workspace_rows)
    print("", flush=True)

    print_section("规则与技能")
    print_section_rows(
        [
            ("Rules", metrics["rules"], ".llgraph/rules + ~/.llgraph/rules"),
            ("Skills", metrics["skills"], ".llgraph/skills + ~/.llgraph/skills"),
            ("Thought", metrics["thought"], "检索无结果扩词等，对齐 Cursor Agent"),
            ("过程展示", metrics["trace"], "/trace all|steps|reply|none"),
            ("Web 搜索", metrics["web"], "Tavily · 仅用户级 agent.json"),
        ],
    )
    print("", flush=True)

    print_section("索引与模型")
    print_section_rows(
        [
            ("Code index", metrics["idx"], ".llgraph/index/"),
            ("LLM 模型", metrics["model"], ""),
            ("向量检索日志", metrics["log"], "/log debug · search.log"),
            ("Index watch", metrics["watch"], ""),
        ],
    )
    print("", flush=True)

    print_section("会话")
    memory_hint = format_storage_location_hint(workspace)
    session_rows: list[tuple[str, str, str]] = [
        ("会话记忆", memory_hint, ""),
        ("MCP", metrics["mcp"], "配置见 .llgraph/mcp.json"),
    ]
    if memory_kind and memory_kind != "none":
        session_rows.insert(
            0,
            ("Checkpointer", memory_kind, "跨轮 LangGraph 状态 + jsonl 落盘"),
        )
    print_section_rows(session_rows)
    print("", flush=True)
    print(sty("exit · /help · /paste · ↑↓ 翻阅历史", "hint"), flush=True)


def build_session_banner_plain(
    *,
    workspace: Path,
    allow_write: bool,
    thread_id: str,
    trace_session: TraceSession,
    watch_active: bool = False,
    web_search_enabled: bool = False,
    mcp_summary: str = "",
    resume_hint: str = "",
    memory_kind: str = "",
) -> str:
    """
    生成启动 Banner 纯文本（无 ANSI，重定向场景）。

    @param workspace 工作区
    @param allow_write 是否可写
    @param thread_id 线程 ID
    @param trace_session 追踪配置
    @return 多行纯文本
    """
    import io

    buf = io.StringIO()
    import sys

    old = sys.stdout
    sys.stdout = buf
    try:
        print_terminal_session_banner(
            workspace=workspace,
            allow_write=allow_write,
            thread_id=thread_id,
            trace_session=trace_session,
            watch_active=watch_active,
            web_search_enabled=web_search_enabled,
            mcp_summary=mcp_summary,
            resume_hint=resume_hint,
            memory_kind=memory_kind,
        )
    finally:
        sys.stdout = old
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())
