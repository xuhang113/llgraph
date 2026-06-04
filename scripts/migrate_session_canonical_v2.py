#!/usr/bin/env python3
"""将会话 messages.jsonl 迁移为 canonical v2，并可清理同工作区其它会话。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from langchain_core.messages import messages_from_dict

from llgraph.context.message_canonical import (
    CANONICAL_FORMAT_VERSION,
    to_canonical_v2_messages,
    validate_canonical_v2_invariants,
)
from llgraph.session.session_file_store import save_session_messages
from llgraph.session.session_meta import load_session_meta, save_session_meta
from llgraph.session.user_storage import session_messages_path, user_sessions_root, workspace_context_dir


def migrate_one(workspace: Path, thread_id: str) -> None:
    """
    迁移单会话 messages.jsonl。

    @param workspace 工作区根
    @param thread_id 会话 ID
    """
    path = session_messages_path(workspace, thread_id)
    if not path.is_file():
        print(f"跳过 {thread_id}: 无 messages.jsonl")
        return
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw = list(messages_from_dict(rows))
    before = len(raw)
    cleaned, report = to_canonical_v2_messages(raw)
    issues = validate_canonical_v2_invariants(cleaned)
    save_session_messages(workspace, thread_id, cleaned)
    meta = load_session_meta(workspace, thread_id)
    meta["messages_format"] = f"canonical_v{CANONICAL_FORMAT_VERSION}"
    save_session_meta(workspace, thread_id, meta)
    print(
        f"{thread_id}: {before} -> {len(cleaned)} 条; "
        f"归档 system {report.archived_system_messages}; "
        f"扁平 AI {report.flattened_ai_messages}; "
        f"校验 {'OK' if not issues else issues}",
    )


def purge_other_sessions(workspace: Path, keep_thread_id: str) -> None:
    """
    删除同工作区除 keep 外的会话目录。

    @param workspace 工作区根
    @param keep_thread_id 保留的 thread_id
    """
    sessions_dir = user_sessions_root(workspace)
    if not sessions_dir.is_dir():
        return
    for child in sessions_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name == keep_thread_id:
            continue
        shutil.rmtree(child)
        print(f"已删除会话目录: {child.name}")
    for p in workspace_context_dir(workspace).glob("cli-*.jsonl"):
        if p.name != f"{keep_thread_id}.jsonl":
            p.unlink(missing_ok=True)
            print(f"已删除遗留文件: {p.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="迁移 canonical v2 并清理会话")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("."),
    )
    parser.add_argument("--thread-id", default="cli-dc2eda6d")
    parser.add_argument(
        "--purge-others",
        action="store_true",
        help="删除该工作区下其它 cli-* 会话",
    )
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    if args.purge_others:
        purge_other_sessions(workspace, args.thread_id)
    migrate_one(workspace, args.thread_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
