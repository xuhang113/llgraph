"""MCP SDK 字段命名兼容回归。

断的是一条语义：**服务端已经执行成功的调用，绝不能被报回模型说失败**。
1.x 是 `isError`，2.x 是 `is_error`，而这些都是 pydantic 模型——
读错名字抛 `AttributeError`，会沿着 except 分支变成一句「MCP 调用失败」，
模型于是重试一次已经生效的写操作。

刻意不依赖可选依赖 `mcp`：CI 只装 `[web]`。
"""

from __future__ import annotations

from llgraph.core.mcp_compat import render_call_result, tool_input_schema


class _PydanticLike:
    """模拟 pydantic 模型：读不存在的字段抛 AttributeError，而不是给 None。"""

    def __init__(self, **fields: object) -> None:
        self.__dict__.update(fields)

    def __getattr__(self, item: str) -> object:
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {item!r}"
        )


def _text_block(text: str) -> _PydanticLike:
    return _PydanticLike(type="text", text=text)


def test_input_schema_reads_both_namings() -> None:
    v2 = _PydanticLike(name="q", input_schema={"properties": {"sql": {}}})
    v1 = _PydanticLike(name="q", inputSchema={"properties": {"sql": {}}})
    assert tool_input_schema(v2) == {"properties": {"sql": {}}}
    assert tool_input_schema(v1) == {"properties": {"sql": {}}}


def test_input_schema_missing_falls_back_to_empty() -> None:
    assert tool_input_schema(_PydanticLike(name="q")) == {}


def test_success_under_mcp_2x_is_not_reported_as_failure() -> None:
    """2.x 只有 is_error：读 isError 会抛 AttributeError，绝不能因此报「调用失败」。"""
    result = _PydanticLike(
        content=[_text_block("42 rows")],
        structured_content=None,
        is_error=False,
    )
    body, is_error = render_call_result(result)
    assert body == "42 rows"
    assert is_error is False


def test_success_under_mcp_1x_is_not_reported_as_failure() -> None:
    result = _PydanticLike(content=[_text_block("42 rows")], isError=False)
    body, is_error = render_call_result(result)
    assert body == "42 rows"
    assert is_error is False


def test_tool_level_error_flag_survives_both_namings() -> None:
    for result in (
        _PydanticLike(content=[_text_block("bad sql")], is_error=True),
        _PydanticLike(content=[_text_block("bad sql")], isError=True),
    ):
        body, is_error = render_call_result(result)
        assert is_error is True
        assert body == "bad sql"


def test_structured_only_result_is_not_rendered_empty() -> None:
    result = _PydanticLike(
        content=[],
        structured_content={"rows": 3},
        is_error=False,
    )
    body, is_error = render_call_result(result)
    assert '"rows": 3' in body
    assert is_error is False
