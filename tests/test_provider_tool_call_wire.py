"""线上协议层：官方入口能跑「一轮对话 + 一次工具调用 + 回灌结果」。

不打真 API：起一个本机 stub server，按各家协议返回 tool_calls，
断言 llgraph 建出来的客户端确实把工具定义发出去了、也确实把 tool_calls 解出来了。
Gemini 走 google-genai SDK、地址不经 llgraph 配置，故本文件不覆盖（见 changelog）。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from llgraph.config.config import ENV_API_BASE_URL, ENV_API_KEY, ENV_MODEL
from llgraph.config.providers import ENV_PROVIDER
from llgraph.core.llm import create_chat_llm
from llgraph.core.llm_settings import set_runtime_model
from llgraph.core.react_graph import _bind_tools_if_needed

_FAKE_KEY = "test-key-not-a-secret"


@tool
def city_time(city: str) -> str:
    """查一个城市的当前时间。"""
    return f"{city} 12:00"


class _StubState:
    """记录 stub server 收到的请求体。"""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []


def _openai_response(state: _StubState) -> dict[str, Any]:
    """第一轮回 tool_calls，第二轮（带 tool 结果）回正文。"""
    body = state.requests[-1]
    has_tool_result = any(m.get("role") == "tool" for m in body.get("messages", []))
    if has_tool_result:
        message: dict[str, Any] = {"role": "assistant", "content": "北京现在 12:00。"}
    else:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "city_time",
                        "arguments": json.dumps({"city": "北京"}),
                    },
                }
            ],
        }
    return {
        "id": "chatcmpl-stub",
        "object": "chat.completion",
        "created": 0,
        "model": body.get("model", "stub"),
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _ollama_response(state: _StubState) -> dict[str, Any]:
    """ollama /api/chat 形态：tool_calls 的 arguments 是对象而不是字符串。"""
    body = state.requests[-1]
    has_tool_result = any(m.get("role") == "tool" for m in body.get("messages", []))
    if has_tool_result:
        message: dict[str, Any] = {"role": "assistant", "content": "北京现在 12:00。"}
    else:
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "city_time", "arguments": {"city": "北京"}}}
            ],
        }
    return {
        "model": body.get("model", "stub"),
        "created_at": "2026-01-01T00:00:00Z",
        "message": message,
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 10,
        "eval_count": 5,
    }


def _make_handler(state: _StubState):
    """
    构造 stub HTTP handler（同时支持 OpenAI /v1 与 ollama /api）。

    @param state 请求记录
    @return handler 类
    """

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:  # 静音
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw.decode("utf-8") or "{}")
            state.requests.append(body)

            if self.path.startswith("/api/chat"):
                payload = _ollama_response(state)
                if body.get("stream"):
                    data = (json.dumps(payload) + "\n").encode("utf-8")
                else:
                    data = json.dumps(payload).encode("utf-8")
            else:
                data = json.dumps(_openai_response(state)).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return _Handler


@pytest.fixture
def stub_server() -> tuple[str, _StubState]:
    """本机 stub server；返回 (base_url, 请求记录)。"""
    state = _StubState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    try:
        yield f"http://{host}:{port}", state
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(autouse=True)
def _clean_runtime_model() -> None:
    set_runtime_model(None)
    yield
    set_runtime_model(None)


def _run_one_tool_round(bound: Any) -> tuple[AIMessage, AIMessage]:
    """
    跑一轮：模型请求工具 → 回灌 ToolMessage → 模型给正文。

    @param bound 已 bind_tools 的 runnable
    @return (带 tool_calls 的响应, 最终响应)
    """
    messages: list[Any] = [HumanMessage(content="北京现在几点？")]
    first = bound.invoke(messages)
    assert first.tool_calls, "第一轮应返回 tool_calls"
    call = first.tool_calls[0]
    assert call["name"] == "city_time"
    assert call["args"] == {"city": "北京"}

    messages.append(first)
    messages.append(
        ToolMessage(
            content=city_time.invoke(call["args"]),
            tool_call_id=call.get("id") or "call_1",
            name="city_time",
        )
    )
    final = bound.invoke(messages)
    return first, final


def test_openai_provider_one_tool_round(
    monkeypatch: pytest.MonkeyPatch,
    stub_server: tuple[str, _StubState],
) -> None:
    """provider=openai：工具定义进 payload，tool_calls 能解出来，回灌后拿到正文。"""
    pytest.importorskip("langchain_openai")
    base_url, state = stub_server
    monkeypatch.delenv(ENV_API_BASE_URL, raising=False)
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    monkeypatch.setenv(ENV_PROVIDER, "openai")
    monkeypatch.setenv("OPENAI_API_KEY", _FAKE_KEY)
    monkeypatch.setenv("LLGRAPH_OPENAI_BASE_URL", f"{base_url}/v1")
    monkeypatch.setenv(ENV_MODEL, "gpt-4.1")

    llm = create_chat_llm(None)
    bound = _bind_tools_if_needed(llm, [city_time])
    _first, final = _run_one_tool_round(bound)

    assert "12:00" in str(final.content)
    sent_tools = state.requests[0].get("tools") or []
    assert [t["function"]["name"] for t in sent_tools] == ["city_time"]
    assert state.requests[0].get("model") == "gpt-4.1"
    assert state.requests[0].get("parallel_tool_calls") is True


def test_ollama_provider_one_tool_round(
    monkeypatch: pytest.MonkeyPatch,
    stub_server: tuple[str, _StubState],
) -> None:
    """provider=ollama：不发 Key、不发 OpenAI 专属参数，一轮工具照样跑通。"""
    pytest.importorskip("langchain_ollama")
    base_url, state = stub_server
    monkeypatch.delenv(ENV_API_BASE_URL, raising=False)
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    monkeypatch.setenv("LLGRAPH_OLLAMA_BASE_URL", base_url)
    monkeypatch.setenv(ENV_MODEL, "qwen3:8b")

    llm = create_chat_llm(None)
    assert getattr(llm, "llgraph_provider") == "ollama"
    bound = _bind_tools_if_needed(llm, [city_time])
    _first, final = _run_one_tool_round(bound)

    assert "12:00" in str(final.content)
    first_req = state.requests[0]
    assert [t["function"]["name"] for t in (first_req.get("tools") or [])] == ["city_time"]
    assert first_req.get("model") == "qwen3:8b"
    # OpenAI / Anthropic 专属参数不能漏到 ollama 请求体里
    assert "parallel_tool_calls" not in first_req
    assert "tool_choice" not in first_req
