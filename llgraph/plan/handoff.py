"""Plan 完成后 handoff 到 Agent 模式。"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from llgraph.session.session_meta import save_session_meta
from llgraph.session.user_storage import session_thread_dir


def create_handoff_session(
    workspace: Path,
    *,
    plan_id: str,
    plan_thread_id: str,
    final_report: str,
    title: str = "",
) -> str:
    """
    创建 Agent 模式 cli 会话并写入 handoff manifest。

    @param workspace 工作区根
    @param plan_id Plan ID
    @param plan_thread_id 原 plan thread
    @param final_report 汇总报告
    @param title 标题
    @return 新 cli thread_id
    """
    cli_thread = f"cli-{uuid.uuid4().hex[:8]}"
    thread_dir = session_thread_dir(workspace, cli_thread)
    thread_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "handoff_from": "plan",
        "plan_id": plan_id,
        "plan_thread_id": plan_thread_id,
        "title": title or f"Handoff from {plan_thread_id}",
        "summary": (final_report or "")[:4000],
        "artifacts_hint": f".llgraph/plans/{plan_id}/",
    }
    manifest_path = thread_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    save_session_meta(
        workspace,
        cli_thread,
        {
            "session_kind": "agent",
            "title": title or f"Plan handoff {plan_id}",
            "handoff_from_plan": plan_thread_id,
            "workspace": str(workspace.expanduser().resolve()),
        },
    )
    return cli_thread


def format_handoff_hint(workspace: Path, cli_thread: str) -> str:
    """
    格式化 handoff 提示。

    @param workspace 工作区根
    @param cli_thread 新 cli thread
    @return 提示文本
    """
    ws = workspace.expanduser().resolve()
    return (
        f"已 handoff 到 Agent 模式会话 {cli_thread}。\n"
        f"继续: llgraph -C {ws} --thread-id {cli_thread}"
    )
