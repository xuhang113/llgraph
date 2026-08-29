"""工具入参纠偏：对齐 Cursor / Claude Code / Codex 的字段名与宽松类型。

网关多模型（Kimi / GLM / DeepSeek / GPT）常按商用 Agent 习惯发参：
``file_path``、``offset``/``limit``、``glob``、``timeout``、``paths`` 写成字符串。
Pydantic 一拒就空转一整轮 LLM。本模块在校验前把别名与类型当场纠正。
"""

from __future__ import annotations

import json
import shlex
from typing import Any

PATH_ALIASES = (
    "file_path",
    "filepath",
    "target_file",
    "targetFile",
    "filePath",
    "relative_workspace_path",
    "file",
)
PATHS_ALIASES = ("files", "file_paths", "filePaths", "targets")
CONTENT_ALIASES = ("contents", "text", "body", "data")
PATTERN_ALIASES = ("query", "search", "regex", "needle")
COMMAND_ALIASES = ("cmd", "shell_command")
CWD_ALIASES = ("cwd", "working_dir", "workdir")
OLD_STRING_ALIASES = ("old_str", "oldString", "old_text", "search")
NEW_STRING_ALIASES = ("new_str", "newString", "new_text", "replace")
REPLACEMENTS_ALIASES = ("edits", "hunks", "changes")
TODOS_ALIASES = ("tasks", "items", "list")
_JSON_START = "{["
_WRAPPER_KEYS = ("arguments", "input", "parameters", "args")

PATH_TOOLS = frozenset(
    {
        "read_file",
        "write_file",
        "append_file",
        "search_replace",
        "list_directory",
        "grep_files",
        "glob_files",
        "search_workspace",
        "search_files",
    }
)
READ_RANGE_TOOLS = frozenset({"read_file", "read_files"})

_CONSUMED_ALIASES = frozenset(
    {
        *PATH_ALIASES,
        *PATHS_ALIASES,
        *CONTENT_ALIASES,
        *PATTERN_ALIASES,
        *COMMAND_ALIASES,
        *CWD_ALIASES,
        *OLD_STRING_ALIASES,
        *NEW_STRING_ALIASES,
        *REPLACEMENTS_ALIASES,
        *TODOS_ALIASES,
        *_WRAPPER_KEYS,
        "offset",
        "limit",
        "timeout",
        "timeout_ms",
        "run_in_background",
        "is_background",
        "background",
        "k",
        "q",
        "glob",
        "include",
        "file_pattern",
        "replaceAll",
        "all",
        "shell_id",
        "job",
        "dir",
        "directory",
        "glob_pattern",
        "file_glob",
        "file_path",
    }
)

_TOOL_KEEP_FIELDS: dict[str, frozenset[str]] = {
    "read_file": frozenset({"path", "start_line", "end_line"}),
    "read_files": frozenset({"paths", "start_line", "end_line"}),
    "list_directory": frozenset({"path"}),
    "glob_files": frozenset({"glob_pattern", "path", "pattern"}),
    "grep_files": frozenset({"pattern", "path", "file_glob", "output_mode", "head_limit"}),
    "search_replace": frozenset(
        {"path", "old_string", "new_string", "replace_all", "replacements"}
    ),
    "write_file": frozenset({"path", "content"}),
    "append_file": frozenset({"path", "content"}),
    "run_shell_command": frozenset({"command", "working_directory", "block_until_ms"}),
    "await_shell": frozenset({"job_id", "block_until_ms", "pattern"}),
    "todo_write": frozenset({"todos", "merge"}),
    "search_code_semantic": frozenset({"query", "top_k", "path_prefix"}),
    "search_code_parallel": frozenset({"query", "top_k", "path_prefix"}),
    "web_search": frozenset({"query", "max_results"}),
    "search_workspace": frozenset({"topic", "keywords", "path", "include_content"}),
    "search_files": frozenset({"keyword", "path", "glob_pattern"}),
}


def maybe_parse_json(raw: object) -> object:
    """
    若值为 JSON 对象/数组字符串则解析，否则原样返回。

    @param raw 原始值
    @return 解析后的对象或原值
    """
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if len(text) < 2 or text[0] not in _JSON_START:
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return raw


def coerce_int(raw: object, default: int | None = None) -> int | None:
    """
    宽松转 int（含数字字符串与 bool）。

    @param raw 原始值
    @param default 无法解析时的缺省；None 表示保持缺失
    @return 整数或 default
    """
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    text = str(raw).strip()
    if not text:
        return default
    try:
        return int(text, 10)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return default


def coerce_bool(raw: object, default: bool | None = None) -> bool | None:
    """
    宽松转 bool。

    @param raw 原始值
    @param default 无法解析时的缺省
    @return 布尔或 default
    """
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return bool(raw)
    text = str(raw).strip().lower()
    if text in {"true", "yes", "y", "on", "1"}:
        return True
    if text in {"false", "no", "n", "off", "0"}:
        return False
    return default


def _fill_alias(data: dict[str, Any], canonical: str, aliases: tuple[str, ...]) -> None:
    if _has_value(data.get(canonical)):
        return
    for key in aliases:
        if key == canonical:
            continue
        if key in data and _has_value(data.get(key)):
            data[canonical] = data[key]
            return


def _has_value(raw: object) -> bool:
    if raw is None:
        return False
    if isinstance(raw, str) and not raw.strip():
        return False
    if isinstance(raw, (list, dict)) and not raw:
        return False
    return True


def _as_dict(raw: object) -> dict[str, Any]:
    parsed = maybe_parse_json(raw)
    if isinstance(parsed, dict):
        return dict(parsed)
    return {}


def _unwrap_nested(data: dict[str, Any]) -> dict[str, Any]:
    """单层 {arguments|input|parameters: {...}} 且顶层无业务字段时展开。"""
    if any(_has_value(data.get(key)) for key in ("path", "paths", "command", "pattern", "query", "todos")):
        return data
    for key in _WRAPPER_KEYS:
        inner = data.get(key)
        parsed = maybe_parse_json(inner)
        if isinstance(parsed, dict) and parsed:
            merged = dict(parsed)
            for extra_key, extra_val in data.items():
                if extra_key == key:
                    continue
                if extra_key not in merged:
                    merged[extra_key] = extra_val
            return merged
    return data


def _parse_json_values(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    for key, value in list(out.items()):
        parsed = maybe_parse_json(value)
        if parsed is not value:
            out[key] = parsed
    return out


def _looks_like_path_item(text: str) -> bool:
    item = text.strip().strip("'\"")
    if not item:
        return False
    return ("/" in item) or ("." in item) or ("\\" in item)


def coerce_path_list(raw: object) -> list[str]:
    """
    把 paths 纠成字符串列表（JSON / 换行 / 逗号 / 单路径）。

    @param raw 模型传入的 paths
    @return 非空路径列表
    """
    parsed = maybe_parse_json(raw)
    if isinstance(parsed, (list, tuple)):
        out: list[str] = []
        for item in parsed:
            item = maybe_parse_json(item)
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
                continue
            if isinstance(item, dict):
                path = item.get("path") or item.get("file_path") or item.get("target_file")
                if isinstance(path, str) and path.strip():
                    out.append(path.strip())
        return out
    if isinstance(parsed, str):
        text = parsed.strip()
        if not text:
            return []
        if "\n" in text:
            return [ln.strip().strip(",") for ln in text.splitlines() if ln.strip()]
        if "," in text:
            parts = [p.strip().strip("'\"") for p in text.split(",") if p.strip()]
            if len(parts) >= 2 and all(_looks_like_path_item(p) for p in parts):
                return parts
        return [text]
    return []


def _apply_path_alias(data: dict[str, Any]) -> None:
    _fill_alias(data, "path", PATH_ALIASES)


def _apply_read_range(data: dict[str, Any]) -> None:
    start = coerce_int(data.get("start_line"))
    end = coerce_int(data.get("end_line"))
    offset = coerce_int(data.get("offset"))
    limit = coerce_int(data.get("limit"))
    if start is None and offset is not None:
        start = 1 if offset <= 0 else offset
        data["start_line"] = start
    if start is None:
        start = 1
    else:
        data["start_line"] = start
    if end is not None:
        data["end_line"] = end
    elif limit is not None and limit > 0:
        data["end_line"] = start + limit - 1


def _apply_paths(data: dict[str, Any]) -> None:
    if not _has_value(data.get("paths")):
        for key in PATHS_ALIASES:
            if _has_value(data.get(key)):
                data["paths"] = data[key]
                break
    paths = coerce_path_list(data.get("paths"))
    if not paths:
        _apply_path_alias(data)
        single = data.get("path")
        if isinstance(single, str) and single.strip():
            paths = [single.strip()]
        elif isinstance(single, list):
            paths = coerce_path_list(single)
    if paths:
        data["paths"] = paths


def _apply_command(data: dict[str, Any]) -> None:
    _fill_alias(data, "command", COMMAND_ALIASES)
    raw = data.get("command")
    if isinstance(raw, list):
        parts = [str(item) for item in raw if str(item).strip() or str(item) == "0"]
        if any(token in {"&&", "||", "|", ";"} for token in parts):
            data["command"] = " ".join(parts)
        else:
            try:
                data["command"] = shlex.join(parts)
            except Exception:
                data["command"] = " ".join(parts)


def _apply_timeout(data: dict[str, Any]) -> None:
    if _has_value(data.get("block_until_ms")):
        value = coerce_int(data.get("block_until_ms"))
        if value is not None:
            data["block_until_ms"] = value
        return
    if _has_value(data.get("timeout_ms")):
        value = coerce_int(data.get("timeout_ms"))
        if value is not None:
            data["block_until_ms"] = value
        return
    if _has_value(data.get("timeout")):
        value = coerce_int(data.get("timeout"))
        if value is None:
            return
        # Claude Code timeout 为毫秒；模型常把 30 当成秒
        data["block_until_ms"] = value if value >= 1000 else value * 1000


def _apply_background(data: dict[str, Any]) -> None:
    for key in ("run_in_background", "is_background", "background"):
        flag = coerce_bool(data.get(key))
        if flag is True:
            data["block_until_ms"] = 0
            return


def _coerce_hunk_item(item: object) -> dict[str, Any] | None:
    parsed = maybe_parse_json(item)
    if not isinstance(parsed, dict):
        return None
    row = dict(parsed)
    _fill_alias(row, "old_string", OLD_STRING_ALIASES)
    _fill_alias(row, "new_string", NEW_STRING_ALIASES)
    if "replace_all" not in row:
        _fill_alias(row, "replace_all", ("replaceAll", "all"))
    flag = coerce_bool(row.get("replace_all"))
    if flag is not None:
        row["replace_all"] = flag
    return row


def _apply_replacements(data: dict[str, Any]) -> None:
    _fill_alias(data, "old_string", OLD_STRING_ALIASES)
    _fill_alias(data, "new_string", NEW_STRING_ALIASES)
    _fill_alias(data, "replace_all", ("replaceAll", "all"))
    flag = coerce_bool(data.get("replace_all"))
    if flag is not None:
        data["replace_all"] = flag
    _fill_alias(data, "replacements", REPLACEMENTS_ALIASES)
    raw = maybe_parse_json(data.get("replacements"))
    if isinstance(raw, list):
        hunks = [hunk for item in raw if (hunk := _coerce_hunk_item(item))]
        if hunks:
            data["replacements"] = hunks


def _apply_todos(data: dict[str, Any]) -> None:
    _fill_alias(data, "todos", TODOS_ALIASES)
    raw = maybe_parse_json(data.get("todos"))
    if isinstance(raw, dict):
        raw = [raw]
    if isinstance(raw, list):
        items: list[dict[str, Any]] = []
        for item in raw:
            parsed = maybe_parse_json(item)
            if isinstance(parsed, str) and parsed.strip():
                items.append({"content": parsed.strip()})
                continue
            if isinstance(parsed, dict):
                row = dict(parsed)
                _fill_alias(row, "content", ("task", "text", "title"))
                items.append(row)
        data["todos"] = items
    flag = coerce_bool(data.get("merge"))
    if flag is not None:
        data["merge"] = flag


def coerce_tool_args(name: str, args: object) -> dict[str, Any]:
    """
    按工具名把模型入参纠成 llgraph schema。

    @param name 工具名
    @param args 原始 args（dict 或 JSON 字符串）
    @return 可交给 Pydantic / 函数的参数 dict
    """
    data = _as_dict(args)
    if not data:
        return {}
    data = _unwrap_nested(data)
    data = _parse_json_values(data)
    tool = (name or "").strip()

    if tool in PATH_TOOLS or tool in READ_RANGE_TOOLS:
        if tool != "read_files":
            _apply_path_alias(data)
    if tool == "read_files":
        _apply_paths(data)
        _apply_read_range(data)
    elif tool in READ_RANGE_TOOLS:
        _apply_read_range(data)

    if tool == "grep_files":
        _fill_alias(data, "pattern", PATTERN_ALIASES)
        _fill_alias(data, "file_glob", ("glob", "include", "file_pattern", "glob_pattern"))
        head = coerce_int(data.get("head_limit"))
        if head is not None:
            data["head_limit"] = head

    if tool == "glob_files":
        _fill_alias(data, "glob_pattern", ("pattern", "glob", "include", "file_glob"))

    if tool == "search_replace":
        _apply_replacements(data)

    if tool in {"write_file", "append_file"}:
        _fill_alias(data, "content", CONTENT_ALIASES)
        content = data.get("content")
        if isinstance(content, list):
            data["content"] = "\n".join(str(item) for item in content)

    if tool == "run_shell_command":
        _apply_command(data)
        _fill_alias(data, "working_directory", CWD_ALIASES)
        _apply_timeout(data)
        _apply_background(data)

    if tool == "await_shell":
        _fill_alias(data, "job_id", ("id", "shell_id", "job"))
        wait = coerce_int(data.get("block_until_ms"))
        if wait is not None:
            data["block_until_ms"] = wait

    if tool == "todo_write":
        _apply_todos(data)

    if tool in {"search_code_semantic", "search_code_parallel"}:
        _fill_alias(data, "query", ("pattern", "search", "q", "text"))
        _fill_alias(data, "path_prefix", ("path", "dir", "directory"))
        top_k = coerce_int(data.get("top_k") if _has_value(data.get("top_k")) else data.get("k"))
        if top_k is not None:
            data["top_k"] = top_k

    if tool == "web_search":
        _fill_alias(data, "query", ("q", "search", "pattern", "text"))
        max_results = coerce_int(data.get("max_results"))
        if max_results is not None:
            data["max_results"] = max_results

    if tool.startswith("mcp__"):
        _apply_path_alias(data)

    _drop_consumed_aliases(data, tool)
    return data


def _drop_consumed_aliases(data: dict[str, Any], tool: str) -> None:
    """去掉已映射的商用别名，避免无 schema 工具因多余 kwargs 失败。"""
    keep = _TOOL_KEEP_FIELDS.get(tool)
    if keep is None:
        for key in PATH_ALIASES:
            if key != "path":
                data.pop(key, None)
        return
    for key in list(data):
        if key in keep:
            continue
        if key in _CONSUMED_ALIASES:
            data.pop(key, None)


def coerce_tool_call(call: Any) -> dict[str, Any]:
    """
    复制 tool_call dict 并把 args 纠偏（不改原对象）。

    @param call LangGraph / OpenAI 风格 tool_call
    @return 新 dict
    """
    if isinstance(call, dict):
        item = dict(call)
        name = str(item.get("name") or "").strip()
        raw_args = item.get("args")
        if raw_args is None:
            raw_args = item.get("arguments")
    else:
        name = str(getattr(call, "name", "") or "").strip()
        raw_args = getattr(call, "args", None)
        if raw_args is None:
            raw_args = getattr(call, "arguments", None)
        item = {
            "name": name,
            "id": getattr(call, "id", None),
            "args": raw_args if isinstance(raw_args, dict) else {},
            "type": "tool_call",
        }
    item["args"] = coerce_tool_args(name, raw_args)
    return item


def wrap_tool_node_with_arg_coerce(inner: Any) -> None:
    """
    包装 ToolNode._run_one / _arun_one：执行前纠偏 args。

    应在 loop_guard / timing / write_serialize **之后**调用（成为最外层），
    使写串行与重复拦截看到的已是规范 path/paths。

    @param inner ToolNode 实例
    """
    if getattr(inner, "_llgraph_arg_coerce_wrapped", False):
        return

    original_run = inner._run_one
    original_arun = inner._arun_one

    def coerced_run_one(call: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        return original_run(coerce_tool_call(call), *args, **kwargs)

    async def coerced_arun_one(call: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        return await original_arun(coerce_tool_call(call), *args, **kwargs)

    inner._run_one = coerced_run_one  # type: ignore[method-assign]
    inner._arun_one = coerced_arun_one  # type: ignore[method-assign]
    inner._llgraph_arg_coerce_wrapped = True  # type: ignore[attr-defined]


def format_tool_validation_error(exc: BaseException) -> str:
    """
    将 Pydantic 校验失败收成短 ToolMessage，避免把整段 schema dump 塞进下一轮。

    @param exc 校验异常
    @return 给模型的错误文案
    """
    text = str(exc or "").strip() or exc.__class__.__name__
    if len(text) > 500:
        text = text[:497] + "..."
    return (
        f"错误: 工具参数无效。{text}\n"
        "已自动兼容 Claude Code / Cursor 别名（file_path、offset/limit、glob、"
        "timeout、run_in_background、paths 字符串）。请改用规范字段后重试。"
    )
