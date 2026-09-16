"""MCP 工具读写判定回归：只读模式该隐藏什么、不该隐藏什么。

断的是**判定结果**，不是判据文案：

- 漏放（写工具在只读会话里可调用）是安全问题：`git_reset` / `git_checkout` /
  `move_file` 会直接毁掉用户未提交的改动
- 误伤（读工具被悄悄摘掉）是可用性问题：模型看不见它，用户也不知道少了什么

第一张表里的工具名与描述是从真实 MCP Server（`mcp-server-git`、`mcp-server-time`、
`mcp-server-fetch`、`@modelcontextprotocol/server-filesystem` / `-memory` /
`-everything` / `-sequential-thinking`）的 `tools/list` 抓下来的，
期望值取服务端自己声明的 `readOnlyHint`。
"""

from __future__ import annotations

import pytest

from llgraph.permissions.mcp import (
    SOURCE_ANNOTATION,
    SOURCE_DEFAULT,
    SOURCE_DESCRIPTION,
    SOURCE_NAME,
    SOURCE_OVERRIDE,
    classify_mcp_tool,
    split_tool_name,
)

# (工具名, 描述首句, 是否写) —— 取自真实 Server 的 tools/list，标注为 readOnlyHint
REAL_TOOLS: tuple[tuple[str, str, bool], ...] = (
    # mcp-server-git：描述是「动词打头 + 无关键词」，老的整段关键词扫描全漏
    ("git_status", "Shows the working tree status", False),
    ("git_diff_unstaged", "Shows changes in the working directory that are not yet staged", False),
    ("git_diff_staged", "Shows changes that are staged for commit", False),
    ("git_diff", "Shows differences between branches or commits", False),
    ("git_log", "Shows the commit logs", False),
    ("git_show", "Shows the contents of a commit", False),
    ("git_branch", "List Git branches", False),
    ("git_commit", "Records changes to the repository", True),
    ("git_add", "Adds file contents to the staging area", True),
    ("git_reset", "Unstages all staged changes", True),
    ("git_checkout", "Switches branches", True),
    ("git_create_branch", "Creates a new branch from an optional base branch", True),
    # @modelcontextprotocol/server-filesystem
    ("read_text_file", "Read the complete contents of a file from the file system as text", False),
    ("read_media_file", "Read a file and return it as a base64-encoded content block", False),
    ("read_multiple_files", "Read the contents of multiple files simultaneously", False),
    ("list_directory", "Get a detailed listing of all files and directories in a specified path", False),
    ("list_directory_with_sizes", "Get a detailed listing of all files and directories, including sizes", False),
    ("directory_tree", "Get a recursive tree view of files and directories as a JSON structure", False),
    ("search_files", "Recursively search for files and directories matching a pattern", False),
    ("get_file_info", "Retrieve detailed metadata about a file or directory", False),
    ("list_allowed_directories", "Returns the list of directories that this server is allowed to access", False),
    ("write_file", "Create a new file or completely overwrite an existing file with new content", True),
    ("edit_file", "Make line-based edits to a text file", True),
    ("create_directory", "Create a new directory or ensure a directory exists", True),
    ("move_file", "Move or rename files and directories", True),
    # @modelcontextprotocol/server-memory
    ("read_graph", "Read the entire knowledge graph", False),
    ("search_nodes", "Search for nodes in the knowledge graph based on a query", False),
    ("open_nodes", "Open specific nodes in the knowledge graph by their names", False),
    ("create_entities", "Create multiple new entities in the knowledge graph", True),
    ("create_relations", "Create multiple new relations between entities in the knowledge graph", True),
    ("add_observations", "Add new observations to existing entities in the knowledge graph", True),
    ("delete_entities", "Delete multiple entities and their associated relations", True),
    ("delete_observations", "Delete specific observations from entities in the knowledge graph", True),
    ("delete_relations", "Delete multiple relations from the knowledge graph", True),
    # mcp-server-time / mcp-server-fetch / sequential-thinking / everything
    ("get_current_time", "Get current time in a specific timezone", False),
    ("convert_time", "Convert time between timezones", False),
    ("fetch", "Fetches a URL from the internet and optionally extracts its contents as markdown", False),
    ("sequentialthinking", "A detailed tool for dynamic and reflective problem-solving through thoughts", False),
    ("echo", "Echoes back the input string", False),
    ("get-env", "Returns all environment variables, helpful for debugging MCP server configuration", False),
    ("get-sum", "Returns the sum of two numbers", False),
    ("get-tiny-image", "Returns a tiny MCP logo image.", False),
    # 老实现在这条上误伤：描述里的 "progress updates" 命中了 update
    ("trigger-long-running-operation", "Demonstrates a long running operation with progress updates.", False),
    ("gzip-file-as-resource", "Compresses a single file using gzip compression", True),
    ("toggle-simulated-logging", "Toggles simulated, random-leveled logging on or off.", True),
)

# 市面常见 Server 的命名形态：读工具名里带写动词（当名词用）、写工具名里带读动词
NAME_SHAPES: tuple[tuple[str, bool], ...] = (
    # 读：动词位第一个词是读动词，后面的 post / commit / patch 是名词
    ("get_post", False),
    ("get_commit", False),
    ("list_posts", False),
    ("list_pull_requests", False),
    ("get_pull_request_files", False),
    ("search_pull_requests", False),
    ("list_starred_repositories", False),
    ("get_issue_comments", False),
    ("get_file_contents", False),
    ("mysql_query", False),
    ("read_query", False),
    ("list_tables", False),
    ("describe_table", False),
    ("browser_take_screenshot", False),
    ("listIssues", False),
    ("resolve-library-id", False),
    # 写：动词位打头或第二位是真动作
    ("create_or_update_file", True),
    ("createIssue", True),
    ("merge_pull_request", True),
    ("update_pull_request_branch", True),
    ("add_issue_comment", True),
    ("slack_post_message", True),
    ("slack_reply_to_thread", True),
    ("fork_repository", True),
    ("push_files", True),
    ("star_repository", True),
    ("write_query", True),
    ("replace_in_file", True),
    ("mkdir", True),
    ("rerun_workflow_run", True),
    ("manage_notification_subscription", True),
    # 写：动词位是软动词且同位置没有读动词
    ("execute_sql", True),
    ("run_command", True),
    ("browser_click", True),
    ("request_copilot_review", True),
    # 名词位才出现写动词：`sub_issue_add` 这类倒装命名
    ("sub_issue_add", True),
)


@pytest.mark.parametrize(("name", "description", "is_write"), REAL_TOOLS)
def test_real_server_tools_without_annotations(
    name: str, description: str, is_write: bool
) -> None:
    """服务端没给标注时，光靠工具名 + 描述起始动词也要判对。"""
    assert classify_mcp_tool(name, description).is_write is is_write


@pytest.mark.parametrize(("name", "description", "is_write"), REAL_TOOLS)
def test_real_server_tools_with_annotations(
    name: str, description: str, is_write: bool
) -> None:
    """标注在时以标注为准（服务端自己声明的语义最可靠）。"""
    access = classify_mcp_tool(
        name, description, annotations={"readOnlyHint": not is_write}
    )
    assert access.is_write is is_write
    assert access.source == SOURCE_ANNOTATION


@pytest.mark.parametrize(("name", "is_write"), NAME_SHAPES)
def test_name_shapes(name: str, is_write: bool) -> None:
    """只给工具名（描述为空）时的判定。"""
    assert classify_mcp_tool(name).is_write is is_write


def test_annotation_beats_name_in_both_directions() -> None:
    """标注可以把名字判定翻过来：两个方向都要生效。"""
    assert classify_mcp_tool("delete_thing", annotations={"readOnlyHint": True}).is_write is False
    assert classify_mcp_tool("list_things", annotations={"readOnlyHint": False}).is_write is True


def test_annotation_snake_case_naming() -> None:
    """MCP SDK 2.x 用 snake_case 字段名。"""
    access = classify_mcp_tool("list_things", annotations={"read_only_hint": False})
    assert access.is_write is True
    assert access.source == SOURCE_ANNOTATION


def test_annotation_object_not_dict() -> None:
    """标注常常是 pydantic 模型而不是 dict。"""

    class _Ann:
        readOnlyHint = True

    assert classify_mcp_tool("delete_thing", annotations=_Ann()).is_write is False


def test_destructive_hint_alone_implies_write() -> None:
    """只声明了 destructiveHint（没有 readOnlyHint）时按写处理。"""
    access = classify_mcp_tool("do_thing", annotations={"destructiveHint": True})
    assert access.is_write is True
    assert access.source == SOURCE_ANNOTATION


def test_non_bool_annotation_falls_through() -> None:
    """标注字段不是布尔（服务端乱填）时不能当真，退回启发式。"""
    access = classify_mcp_tool(
        "delete_thing", annotations={"readOnlyHint": "yes"}
    )
    assert access.is_write is True
    assert access.source == SOURCE_NAME


def test_override_wins_over_annotation() -> None:
    """人工覆盖压过一切：标注也可能是错的。"""
    assert (
        classify_mcp_tool(
            "list_things",
            annotations={"readOnlyHint": True},
            write_tools=["list_things"],
        ).is_write
        is True
    )
    assert (
        classify_mcp_tool(
            "delete_things",
            annotations={"readOnlyHint": False},
            read_tools=["delete_things"],
        ).is_write
        is False
    )


def test_override_glob_and_case_insensitive() -> None:
    access = classify_mcp_tool("Run_Report", read_tools=["run_*"])
    assert access.is_write is False
    assert access.source == SOURCE_OVERRIDE
    assert access.detail == "run_*"


def test_override_write_wins_when_both_match() -> None:
    """两张表都命中时按写处理：宁可多隐藏一个，也不要放出真写工具。"""
    assert (
        classify_mcp_tool("sync_all", read_tools=["*"], write_tools=["sync_*"]).is_write
        is True
    )


def test_override_ignores_bad_config_shapes() -> None:
    """配置写成字符串 / 数字 / None 不能让判定崩掉。"""
    for bad in (None, "", "sync_all", 3, [None, 7], [" "]):
        assert classify_mcp_tool("sync_all", read_tools=bad).is_write is True


def test_description_only_reads_leading_verb() -> None:
    """描述只看第一句起始动词：整段扫关键词是老实现误伤的来源。"""
    # 「后面的句子在讲写」不算写：这是在说明另一个工具怎么用
    access = classify_mcp_tool(
        "thing_probe",
        "Returns the current value. Use create_or_update_thing to write it.",
    )
    assert access.is_write is False
    # 起始动词是写动词才算写
    assert classify_mcp_tool("thing_probe", "Records the value").is_write is True


def test_description_skips_leading_adverb() -> None:
    access = classify_mcp_tool("files_probe", "Recursively search for files")
    assert access.is_write is False
    assert access.source == SOURCE_DESCRIPTION


def test_unknown_tool_defaults_to_read() -> None:
    """判不出来默认按读：否则只读会话的外部能力会被大面积摘掉。"""
    access = classify_mcp_tool("frobnicate", "A tool.")
    assert access.is_write is False
    assert access.source == SOURCE_DEFAULT


def test_name_verdict_beats_description() -> None:
    """名字判得出来就不看描述：描述往往在讲「配合哪个工具用」。"""
    access = classify_mcp_tool("delete_thing", "Shows what would be affected")
    assert access.is_write is True
    assert access.source == SOURCE_NAME


def test_name_tokens_are_not_stemmed() -> None:
    """工具名不做词干还原：`unstaged` 还原成 `unstage` 会把只读 diff 判成写。"""
    assert split_tool_name("git_diff_unstaged") == ["git", "diff", "unstaged"]
    assert classify_mcp_tool("git_diff_unstaged").is_write is False


def test_split_tool_name_handles_camel_and_separators() -> None:
    assert split_tool_name("createOrUpdateFile") == ["create", "or", "update", "file"]
    assert split_tool_name("get-library-docs") == ["get", "library", "docs"]
    assert split_tool_name("slack.post.message") == ["slack", "post", "message"]
    assert split_tool_name("") == []


def test_empty_name_and_description() -> None:
    access = classify_mcp_tool("", "")
    assert access.is_write is False
    assert access.source == SOURCE_DEFAULT


def test_reason_is_reported_for_observability() -> None:
    """判据要能打出来：判错时用户得看得见是哪一层、命中了什么。"""
    assert classify_mcp_tool("git_commit").reason() == f"{SOURCE_NAME}:commit"
    assert (
        classify_mcp_tool("x", annotations={"readOnlyHint": False}).reason()
        == f"{SOURCE_ANNOTATION}:readOnlyHint=false"
    )
    assert classify_mcp_tool("frobnicate").reason() == SOURCE_DEFAULT


# --------------------------------------------- 加载期过滤：隐藏了什么、说了没说


class _FakeTool:
    """模拟 MCP `Tool`：读不存在的字段抛 AttributeError，而不是给 None。"""

    def __init__(
        self, name: str, description: str = "", annotations: object = None
    ) -> None:
        self.name = name
        self.description = description
        if annotations is not None:
            self.annotations = annotations

    def __getattr__(self, item: str) -> object:
        raise AttributeError(item)


def _runtime(tools: list[_FakeTool], **cfg: object) -> object:
    """造一个不连子进程的 runtime：工具表直接塞进去，判定走生产代码。"""
    from llgraph.config.mcp_config import McpServerConfig
    from llgraph.core.mcp_compat import tool_annotations, tool_description
    from llgraph.core.mcp_tools import _McpServerRuntime

    config = McpServerConfig(
        name="probe", command="true", args=[], env={}, cwd=None, enabled=True, **cfg
    )
    runtime = _McpServerRuntime(config, timeout_sec=5.0)
    runtime._tools = list(tools)
    for item in tools:
        runtime._tool_desc[item.name] = tool_description(item)
        ann = tool_annotations(item)
        if ann is not None:
            runtime._tool_annotations[item.name] = ann
    return runtime


def _registry(runtime: object, *, allow_write: bool) -> object:
    from llgraph.config.mcp_config import McpSettings
    from llgraph.core.mcp_tools import McpToolRegistry

    settings = McpSettings(
        servers=(runtime.config,),
        timeout_sec=5.0,
        allow_write_tools=allow_write,
        config_source="(test)",
    )
    registry = McpToolRegistry(settings)
    registry._runtimes["probe"] = runtime
    registry._langchain_tools.extend(
        registry._build_tools_for_server("probe", runtime)
    )
    return registry


_PROBE_TOOLS = [
    _FakeTool("git_status", "Shows the working tree status"),
    _FakeTool("git_commit", "Records changes to the repository"),
    _FakeTool("mysql_query", "Run a SQL query. Supports SELECT/INSERT/UPDATE/DELETE."),
    _FakeTool("sync_now", "Keeps both sides in step"),
]


def test_readonly_hides_write_tools_and_keeps_read_tools() -> None:
    registry = _registry(_runtime(_PROBE_TOOLS), allow_write=False)
    names = [t.name for t in registry.get_tools()]
    assert names == ["mcp__probe__git_status", "mcp__probe__mysql_query"]
    assert registry.hidden_write_tools == {"probe": ["git_commit", "sync_now"]}


def test_hidden_tools_are_named_in_summary() -> None:
    """隐藏必须可见：判错时用户得能看出少了哪个工具、怎么要回来。"""
    registry = _registry(_runtime(_PROBE_TOOLS), allow_write=False)
    summary = registry.summary()
    assert "git_commit" in summary and "sync_now" in summary
    assert "read_tools" in summary
    assert "git_status" not in summary


def test_allow_write_hides_nothing() -> None:
    registry = _registry(_runtime(_PROBE_TOOLS), allow_write=True)
    assert len(registry.get_tools()) == len(_PROBE_TOOLS)
    assert registry.hidden_write_tools == {}


def test_config_read_tools_override_restores_hidden_tool() -> None:
    """启发式判错时，用户不必为一个工具把整台 Server 的写工具全打开。"""
    runtime = _runtime(_PROBE_TOOLS, read_tools=("sync_*",))
    registry = _registry(runtime, allow_write=False)
    names = [t.name for t in registry.get_tools()]
    assert "mcp__probe__sync_now" in names
    assert registry.hidden_write_tools == {"probe": ["git_commit"]}


def test_config_write_tools_override_hides_extra_tool() -> None:
    runtime = _runtime(_PROBE_TOOLS, write_tools=("mysql_query",))
    registry = _registry(runtime, allow_write=False)
    assert [t.name for t in registry.get_tools()] == ["mcp__probe__git_status"]


def test_server_annotation_overrides_name_heuristic_at_load() -> None:
    """服务端说自己只读就按只读加载，别按名字猜。"""
    tools = [_FakeTool("delete_stale_cache", "", {"readOnlyHint": True})]
    registry = _registry(_runtime(tools), allow_write=False)
    assert [t.name for t in registry.get_tools()] == ["mcp__probe__delete_stale_cache"]


def test_replay_decision_matches_filter_verdict() -> None:
    """过滤与「重连后能不能重放」必须用同一个判定，否则被隐藏的写工具反而会被重放。"""
    tools = [_FakeTool("list_things", "Lists things", {"readOnlyHint": False})]
    runtime = _runtime(tools)
    assert runtime.tool_access("list_things").is_write is True
    registry = _registry(runtime, allow_write=False)
    assert registry.get_tools() == []


def test_tool_annotations_compat_reads_missing_field_safely() -> None:
    """1.x 早期没有 annotations 字段：取不到要退回启发式，不能抛。"""
    from llgraph.core.mcp_compat import tool_annotations

    assert tool_annotations(_FakeTool("x", "y")) is None
    assert tool_annotations(_FakeTool("x", "y", {"readOnlyHint": True}) ) == {
        "readOnlyHint": True
    }
