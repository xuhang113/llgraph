#!/usr/bin/env python3
"""一次性包结构整理：按业务域移动 llgraph 模块并更新 import。"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "llgraph"

# 模块名 -> 新子包（llgraph.<pkg>.<module>）
PACKAGE_MAP: dict[str, str] = {
    # core — Agent 编排、LLM、工具
    "agent": "core",
    "agent_config": "core",
    "agent_session": "core",
    "checkpointer_factory": "core",
    "llm": "core",
    "llm_settings": "core",
    "gateway_models": "core",
    "gateway_kimi_patch": "core",
    "model_context_window": "core",
    "prompt_cache": "core",
    "prompt_cache_settings": "core",
    "tools": "core",
    "filesystem_tools": "core",
    "shell_tools": "core",
    "web_search_tools": "core",
    "mcp_tools": "core",
    "code_index_tools": "core",
    "tool_list": "core",
    "workspace": "core",
    "write_failure_tracker": "core",
    # context — 上下文、消息、压缩
    "context_session": "context",
    "context_settings": "context",
    "context_builder": "context",
    "context_compressor": "context",
    "context_dispatch_window": "context",
    "context_message_split": "context",
    "context_spill": "context",
    "context_stats": "context",
    "incremental_context": "context",
    "conversation_anchor": "context",
    "message_canonical": "context",
    "message_dispatch_profile": "context",
    "message_normalize": "context",
    "chat_history_repair": "context",
    "runtime_context": "context",
    # session — 会话持久化与生命周期
    "session_file_store": "session",
    "session_registry": "session",
    "session_meta": "session",
    "session_manifest": "session",
    "session_switch": "session",
    "session_delete": "session",
    "session_edits": "session",
    "session_web_search": "session",
    "session_history_search": "session",
    "session_history_tools": "session",
    "session_write_mode": "session",
    "user_storage": "session",
    # config — 配置与 settings
    "config": "config",
    "workspace_config": "config",
    "mcp_config": "config",
    "catalog_paths": "config",
    "logging_settings": "config",
    "edit_settings": "config",
    "shell_settings": "config",
    "web_search_settings": "config",
    "survey_settings": "config",
    # loaders — Rules / Skills / Commands 加载
    "rules_loader": "loaders",
    "skills_loader": "loaders",
    "commands_loader": "loaders",
    "thought_loader": "loaders",
    # commands — 斜杠命令实现
    "meta_commands": "commands",
    "review_command": "commands",
    "help_text": "commands",
    # survey — 交互确认
    "survey_prompt": "survey",
    "edit_confirm": "survey",
    # display — 追踪、日志、终端样式
    "trace_display": "display",
    "execution_log": "display",
    "terminal_style": "display",
    "log_retention": "display",
    # cli — 子命令
    "index_cli": "cli",
    "search_cli": "cli",
    "markdowns_index": "cli",
    "search_terms": "cli",
}

KEEP_AT_ROOT = {"main.py", "__main__.py", "__init__.py"}

PACKAGE_DOCS: dict[str, str] = {
    "core": "Agent 编排、Gateway LLM、工具注册与实现。",
    "context": "上下文构建、消息规范化、压缩与出站窗口。",
    "session": "会话持久化、切换、编辑记录与历史检索。",
    "config": "环境变量、agent.json 与各 feature settings。",
    "loaders": "Rules、Skills、Commands、Thought 加载。",
    "commands": "斜杠命令（/review、/trace 等）实现。",
    "survey": "交互式 survey 与编辑确认。",
    "display": "过程追踪、执行日志与终端样式。",
    "cli": "llgraph index / search 等 CLI 子命令。",
}


def _new_import_path(module: str) -> str:
    pkg = PACKAGE_MAP[module]
    return f"llgraph.{pkg}.{module}"


def _ensure_package_init(pkg_dir: Path, doc: str) -> None:
    init = pkg_dir / "__init__.py"
    if not init.exists():
        init.write_text(f'"""{doc}"""\n', encoding="utf-8")


def move_modules() -> None:
    for module, pkg in PACKAGE_MAP.items():
        src = PKG / f"{module}.py"
        if not src.exists():
            raise FileNotFoundError(src)
        dest_dir = PKG / pkg
        dest_dir.mkdir(parents=True, exist_ok=True)
        _ensure_package_init(dest_dir, PACKAGE_DOCS.get(pkg, pkg))
        dest = dest_dir / f"{module}.py"
        if dest.exists():
            raise FileExistsError(dest)
        shutil.move(str(src), str(dest))


def _replace_imports(text: str) -> str:
    """将 llgraph.<mod> 替换为新路径（跳过已有子包路径）。"""
    for module in sorted(PACKAGE_MAP, key=len, reverse=True):
        new_path = _new_import_path(module)
        old_plain = f"llgraph.{module}"
        if old_plain == new_path:
            continue
        # 已是 llgraph.core.agent 等形式则跳过
        text = re.sub(
            rf"\bllgraph\.{re.escape(module)}\b(?!\.|\w)",
            new_path,
            text,
        )
    return text


def rewrite_all_imports() -> None:
    targets: list[Path] = []
    for base in (PKG, ROOT / "scripts"):
        if not base.exists():
            continue
        targets.extend(base.rglob("*.py"))
    for path in targets:
        if path.name == "reorganize_packages.py":
            continue
        raw = path.read_text(encoding="utf-8")
        updated = _replace_imports(raw)
        if updated != raw:
            path.write_text(updated, encoding="utf-8")


def main() -> None:
    print("1/3 移动模块文件 …")
    move_modules()
    print("2/3 重写 import …")
    rewrite_all_imports()
    print("完成。请运行: pip install -e . && python -c 'from llgraph.main import main'")


if __name__ == "__main__":
    main()
