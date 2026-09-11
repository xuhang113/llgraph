"""网关 LLM 流式调用的瞬时故障分类与重试节奏。

出站走的是 `.stream()`：一旦响应头已经回来，SDK 自带的 `max_retries` 就不再兜底，
中途被网关掐断（502 / connection reset / incomplete chunked read）会直接把整轮
ReAct 打掉——哪怕前面已经跑完十几个工具。这里给出「哪些错该重试」和「等多久」，
由 `react_invoke` 在流层面重放整次调用。

判定只看异常类名、HTTP 状态码与消息文本，不 import anthropic / httpx，
避免把重 SDK 拉进启动早期路径。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BASE_DELAY_SEC = 1.0
DEFAULT_MAX_DELAY_SEC = 20.0

_MAX_CAUSE_DEPTH = 6
_MESSAGE_SCAN_LIMIT = 600

# 网关侧压不住的瞬时状态码；409/425 属于并发重放，也可以直接再来一次
_TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529})
# 请求本身有问题，重试只会再烧一次 prompt
_FATAL_STATUS = frozenset({400, 401, 402, 403, 404, 405, 413, 414, 422})

_TRANSIENT_TYPES = frozenset(
    {
        # anthropic / openai SDK
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
        "ServiceUnavailableError",
        "OverloadedError",
        "ConflictError",
        # httpx / httpcore
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "ReadError",
        "WriteError",
        "RemoteProtocolError",
        "LocalProtocolError",
        "ProtocolError",
        "IncompleteRead",
        "ChunkedEncodingError",
        # 标准库
        "TimeoutError",
        "ConnectionError",
        "ConnectionResetError",
        "ConnectionAbortedError",
        "IncompleteReadError",
    }
)

_FATAL_TYPES = frozenset(
    {
        "UserCancelledError",
        "ThinkingStreamTimeoutError",
        "KeyboardInterrupt",
        "GeneratorExit",
        "SystemExit",
        "MemoryError",
        "BadRequestError",
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
        "UnprocessableEntityError",
    }
)

# 网关常把上游故障塞进 200 的 SSE body，langchain 抛出来就是个普通 Exception
_TRANSIENT_PHRASES = (
    "overloaded",
    "rate limit",
    "rate_limit",
    "too many requests",
    "server disconnected",
    "peer closed connection",
    "connection reset",
    "connection aborted",
    "connection closed",
    "incomplete chunked read",
    "incomplete read",
    "bad gateway",
    "gateway timeout",
    "service unavailable",
    "temporarily unavailable",
    "internal server error",
    "upstream connect error",
    "upstream request timeout",
    "timed out",
    "eof occurred",
)

_FATAL_PHRASES = (
    "context length",
    "context_length",
    "maximum context",
    "prompt is too long",
    "too many tokens",
    "invalid_request_error",
    "invalid api key",
    "authentication_error",
    "permission_error",
    "credit balance",
    "insufficient",
    "model not found",
    "unsupported",
)


@dataclass(frozen=True)
class LlmStreamRetryPolicy:
    """单次 LLM 流式调用的重放预算。"""

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    base_delay_sec: float = DEFAULT_BASE_DELAY_SEC
    max_delay_sec: float = DEFAULT_MAX_DELAY_SEC

    @property
    def enabled(self) -> bool:
        return self.max_attempts > 1


def _status_of(exc: BaseException) -> int | None:
    for holder, attr in ((exc, "status_code"), (getattr(exc, "response", None), "status_code")):
        if holder is None:
            continue
        raw = getattr(holder, attr, None)
        if isinstance(raw, bool):
            continue
        if isinstance(raw, int):
            return raw
        if isinstance(raw, str) and raw.strip().isdigit():
            return int(raw.strip())
    return None


def _cause_chain(exc: BaseException) -> list[BaseException]:
    """展开 __cause__ / __context__（SDK 常把 httpx 错误包一层）。"""
    chain: list[BaseException] = []
    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    while pending and len(chain) < _MAX_CAUSE_DEPTH:
        current = pending.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        chain.append(current)
        for nxt in (current.__cause__, current.__context__):
            if isinstance(nxt, BaseException) and id(nxt) not in seen:
                pending.append(nxt)
    return chain


def _text_of(exc: BaseException) -> str:
    try:
        return str(exc)[:_MESSAGE_SCAN_LIMIT].lower()
    except Exception:
        return ""


def classify_stream_failure(exc: BaseException) -> str | None:
    """
    判断一次流式失败是否值得原样重放。

    @param exc 流式过程中抛出的异常
    @return 瞬时故障原因标签（写进 run_log）；None 表示不该重试
    """
    chain = _cause_chain(exc)

    # 致命信号优先：包装层看着像网络错，内层其实是 400 的情况很常见
    for item in chain:
        if type(item).__name__ in _FATAL_TYPES:
            return None
        status = _status_of(item)
        if status is not None and status in _FATAL_STATUS:
            return None
        text = _text_of(item)
        if any(phrase in text for phrase in _FATAL_PHRASES):
            return None

    for item in chain:
        status = _status_of(item)
        if status is not None and status in _TRANSIENT_STATUS:
            return f"http_{status}"
    for item in chain:
        name = type(item).__name__
        if name in _TRANSIENT_TYPES:
            return name
    for item in chain:
        text = _text_of(item)
        for phrase in _TRANSIENT_PHRASES:
            if phrase in text:
                return phrase.replace(" ", "_")
    return None


def compute_retry_delay(
    attempt: int,
    policy: LlmStreamRetryPolicy,
    *,
    jitter: float | None = None,
) -> float:
    """
    指数退避 + 抖动（多个并发会话不要同时撞回网关）。

    @param attempt 已失败次数，从 1 开始
    @param policy 重放预算
    @param jitter 0~1 的抖动比例；None 表示随机
    @return 睡眠秒数
    """
    step = max(1, int(attempt))
    delay = policy.base_delay_sec * (2 ** (step - 1))
    delay = min(delay, policy.max_delay_sec)
    ratio = random.random() if jitter is None else max(0.0, min(1.0, float(jitter)))
    return round(delay * (1.0 + 0.25 * ratio), 3)


def _coerce_policy(raw: Any) -> LlmStreamRetryPolicy:
    if not isinstance(raw, dict):
        return LlmStreamRetryPolicy()

    def _num(key: str, fallback: float, low: float, high: float) -> float:
        try:
            return max(low, min(high, float(raw.get(key, fallback))))
        except (TypeError, ValueError):
            return fallback

    try:
        attempts = max(1, min(6, int(raw.get("max_attempts", DEFAULT_MAX_ATTEMPTS))))
    except (TypeError, ValueError):
        attempts = DEFAULT_MAX_ATTEMPTS
    base = _num("base_delay_sec", DEFAULT_BASE_DELAY_SEC, 0.05, 30.0)
    ceiling = _num("max_delay_sec", DEFAULT_MAX_DELAY_SEC, base, 120.0)
    return LlmStreamRetryPolicy(
        max_attempts=attempts,
        base_delay_sec=base,
        max_delay_sec=max(base, ceiling),
    )


def resolve_llm_stream_retry_policy(workspace: Path | None = None) -> LlmStreamRetryPolicy:
    """
    读取 `.llgraph/agent.json` → `llm.stream_retry`。

    @param workspace 工作区根；None 用默认
    @return 重放预算
    """
    if workspace is None:
        return LlmStreamRetryPolicy()
    try:
        from llgraph.config.edit_settings import load_agent_config

        cfg = load_agent_config(workspace)
    except Exception:
        return LlmStreamRetryPolicy()
    llm_cfg = cfg.get("llm") if isinstance(cfg, dict) else None
    if not isinstance(llm_cfg, dict):
        return LlmStreamRetryPolicy()
    return _coerce_policy(llm_cfg.get("stream_retry"))
