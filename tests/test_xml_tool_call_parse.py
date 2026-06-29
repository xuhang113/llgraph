"""XML <tool_call> 入站解析单测。"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from llgraph.adapters.inbound import normalize_ai_response, parse_xml_tool_calls, strip_inbound_tool_call_markup
from llgraph.adapters.inbound.profile import InboundAdapterProfile
from llgraph.adapters.inbound.xml_tool_call import normalize_xml_tool_calls
from llgraph.core.agent_turn import route_after_agent


def test_parse_qwen_xml_tool_call() -> None:
    raw = (
        "【规划】并行检索 markdown 相关代码。\n"
        "<tool_call>\n"
        "<function=glob_files>\n"
        "<parameter=path>\n"
        ".\n"
        "</parameter>\n"
        "<parameter=pattern>\n"
        "**/*markdown*\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    calls, text = parse_xml_tool_calls(raw)
    assert text == "【规划】并行检索 markdown 相关代码。"
    assert len(calls) == 1
    assert calls[0]["name"] == "glob_files"
    assert calls[0]["args"]["path"] == "."
    assert calls[0]["args"]["pattern"] == "**/*markdown*"


def test_parse_arg_key_value_tool_call() -> None:
    raw = (
        "【规划】先搜文档渲染逻辑。\n"
        "<tool_call>\n"
        "search_code_parallel\n"
        "<arg_key>query</arg_key>\n"
        '<arg_value>"markdown 渲染兼容"</arg_value>\n'
        "</tool_call>"
    )
    calls, text = parse_xml_tool_calls(raw)
    assert text == "【规划】先搜文档渲染逻辑。"
    assert len(calls) == 1
    assert calls[0]["name"] == "search_code_parallel"
    assert calls[0]["args"]["query"] == "markdown 渲染兼容"


def test_parse_json_inside_tool_call() -> None:
    raw = (
        '<tool_call>{"name": "grep_files", "arguments": {"pattern": "markdown", "path": "."}}'
        "</tool_call>"
    )
    calls, text = parse_xml_tool_calls(raw)
    assert text == ""
    assert calls[0]["name"] == "grep_files"
    assert calls[0]["args"] == {"pattern": "markdown", "path": "."}


def test_normalize_xml_routes_to_tools() -> None:
    msg = AIMessage(
        content=(
            "【规划】探测目录。\n"
            "<tool_call>\n"
            "<function=list_directory>\n"
            "<parameter=path>\n"
            ".\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
    )
    repaired, changed = normalize_xml_tool_calls(msg)
    assert changed
    assert len(repaired.tool_calls) == 1
    assert repaired.tool_calls[0]["name"] == "list_directory"
    state = {
        "messages": [AIMessage(content="q"), repaired],
        "remaining_steps": 10,
    }
    assert route_after_agent(state) == "tools"


def test_normalize_ai_response_rejects_unstructured_xml() -> None:
    sample = (
        "<tool_call>\n"
        "<function=read_file>\n"
        "<parameter=path>\n"
        "README.md\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    msg = AIMessage(content=sample)
    import pytest
    from llgraph.adapters.inbound import UnstructuredToolCallError, normalize_ai_response

    with pytest.raises(UnstructuredToolCallError):
        normalize_ai_response(
            msg,
            None,
            "claude-sonnet-4-6",
            profile=InboundAdapterProfile(parse_kimi_native_tool_tokens=False),
        )


def test_parse_deepseek_anthropic_xml_tool_calls() -> None:
    raw = (
        "【规划】先并行 grep。\n"
        "<tool_calls>\n"
        '<tool_call name="grep_files">\n'
        "<tool_call_name>grep_files</tool_call_name>\n"
        "<tool_call_id>toolu_bdrk_01</tool_call_id>\n"
        '<tool_call_args>{"pattern": "export|导出", "path": ".", "include": "*.java"}</tool_call_args>\n'
        "</tool_call>\n"
        '<tool_call name="grep_files">\n'
        "<tool_call_name>grep_files</tool_call_name>\n"
        "<tool_call_id>toolu_bdrk_02</tool_call_id>\n"
        '<tool_call_args>{"pattern": "orderId", "path": "."}</tool_call_args>\n'
        "</tool_call>\n"
        "</tool_calls>"
    )
    calls, text = parse_xml_tool_calls(raw)
    assert text == "【规划】先并行 grep。"
    assert len(calls) == 2
    assert calls[0]["name"] == "grep_files"
    assert calls[0]["id"] == "toolu_bdrk_01"
    assert calls[0]["args"]["pattern"] == "export|导出"
    assert calls[1]["args"]["pattern"] == "orderId"


def test_normalize_deepseek_xml_raises_without_structured_calls() -> None:
    msg = AIMessage(
        content=(
            "【规划】检索。\n"
            "<tool_calls>\n"
            '<tool_call name="grep_files">\n'
            "<tool_call_name>grep_files</tool_call_name>\n"
            '<tool_call_args>{"pattern": "export", "path": "."}</tool_call_args>\n'
            "</tool_call>\n"
            "</tool_calls>"
        )
    )
    import pytest
    from llgraph.adapters.inbound import UnstructuredToolCallError, normalize_ai_response

    with pytest.raises(UnstructuredToolCallError):
        normalize_ai_response(msg, None, "deepseek-v4-pro")


def test_parse_deepseek_argument_tag_tool_calls() -> None:
    raw = (
        "<tool_calls>\n"
        '<tool_call name="grep_files">\n'
        '<argument name="pattern" string="true">orderId.*status</argument>\n'
        '<argument name="path" string="true">.</argument>\n'
        "</tool_call>\n"
        '<tool_call name="read_file">\n'
        '<argument name="path" string="true">README.md</argument>\n'
        '<argument name="start_line" string="false">1</argument>\n'
        '<argument name="end_line" string="false">80</argument>\n'
        "</tool_call>\n"
        "</tool_calls>"
    )
    calls, text = parse_xml_tool_calls(raw)
    assert text == ""
    assert len(calls) == 2
    assert calls[0]["name"] == "grep_files"
    assert calls[0]["args"]["pattern"] == "orderId.*status"
    assert calls[1]["args"]["start_line"] == 1


def test_strip_inbound_tool_call_markup_keeps_plan_prefix() -> None:
    raw = "【规划】说明\n<tool_call>search_code_parallel\n</tool_call>"
    assert strip_inbound_tool_call_markup(raw) == "【规划】说明"
