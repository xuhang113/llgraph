"""manage_memory / search_memory 工具。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import StructuredTool

from llgraph.memory.recall import format_agent_memories_block, recall_memories, record_memory_hits
from llgraph.memory.scheduler import touch_memory_workspace
from llgraph.memory.settings import resolve_memory_settings
from llgraph.memory.write import delete_memory, upsert_memory


def create_memory_tools(workspace_root: Path) -> list:
    """
    创建长期记忆工具。

    @param workspace_root 工作区根
    @return Tool 列表
    """
    root = workspace_root.expanduser().resolve()
    settings = resolve_memory_settings(root)
    if not settings.enabled:
        return []

    def manage_memory(
        action: str,
        content: str = "",
        kind: str = "pref",
        memory_id: str = "",
    ) -> str:
        """
        管理本会话工作区的长期记忆（偏好/事实/做法）。

        **应调用 create/update（不必等用户说「记住」）**：
        - 用户给出可复用的回答/协作规范（如「说明要写全场景」「普通直播与随手播要分开」）；
        - 指出当前回答不完整/会误导，并给出以后应怎么写；
        - 明确纠正偏好、稳定事实，或可跨会话的通用做法。
        先 manage_memory 记一条抽象规范，再继续答；勿把调研长文、YAML 全文、单次任务步骤写入。
        未核实的「应该是某字段=某业务」可只记写法规范，核实后再记 fact。delete 删除一条。

        @param action create|update|delete
        @param content 记忆正文（create/update）
        @param kind pref|fact|proc（create/update）
        @param memory_id 删除或更新时指定 ID
        """
        touch_memory_workspace(root)
        act = (action or "").strip().lower()
        if act == "delete":
            if not memory_id.strip():
                return "manage_memory: delete 需要 memory_id。"
            report = delete_memory(root, memory_id.strip())
            from llgraph.memory.trace_emit import emit_memory_write_trace_step

            emit_memory_write_trace_step(report)
            return f"已删除 memory_id={memory_id}"
        if act not in ("create", "update"):
            return "manage_memory: action 须为 create|update|delete。"
        if not content.strip():
            return "manage_memory: content 不能为空。"
        k = (kind or "pref").strip().lower()
        if k not in ("pref", "fact", "proc"):
            k = "pref"
        report = upsert_memory(
            root,
            content=content,
            kind=k,
            memory_id=memory_id.strip() or None,
            source="hot_tool",
        )
        from llgraph.memory.trace_emit import emit_memory_write_trace_step

        emit_memory_write_trace_step(report)
        return (
            f"已{report.action} memory_id={report.memory_id} kind={report.kind}"
            + (f" 替换={report.replaced_ids}" if report.replaced_ids else "")
        )

    def search_memory(query: str, top_k: int = 10) -> str:
        """
        检索本会话工作区长期记忆（偏好、事实、通用做法）。

        用户问「记得吗/我的偏好」或自动召回不足时使用；非会话情节（用 search_session_history）。

        @param query 检索问句或关键词
        @param top_k 条数，默认 10
        """
        touch_memory_workspace(root)
        q = (query or "").strip()
        if not q:
            return "search_memory: query 不能为空。"
        try:
            k = max(1, min(30, int(top_k)))
        except (TypeError, ValueError):
            k = settings.search_tool_top_k
        hits, report = recall_memories(root, q, top_k=k, for_tool=True)
        if not hits:
            return f"未找到与「{q}」相关的长期记忆。"
        record_memory_hits(root, hits)
        block = format_agent_memories_block(
            hits,
            excerpt_chars=settings.memory_inject_excerpt_chars,
            max_tokens=settings.max_inject_tokens * 2,
        )
        lines = [block, "", f"（共 {len(hits)} 条，可用 memory_id 配合 manage_memory delete 管理）"]
        for hit in hits:
            mid = hit.memory_id[:8] + "…"
            lines.append(f"- id={mid} score={hit.score:.2f} [{hit.kind}]")
        return "\n".join(lines)

    return [
        StructuredTool.from_function(
            func=manage_memory,
            name="manage_memory",
            description=manage_memory.__doc__ or "",
        ),
        StructuredTool.from_function(
            func=search_memory,
            name="search_memory",
            description=search_memory.__doc__ or "",
        ),
    ]
