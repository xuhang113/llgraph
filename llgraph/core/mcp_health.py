"""MCP 传输层故障分类与失败回灌文案。

区分两类失败，因为处置完全不同：

- **工具自己报错**（SQL 语法错、参数不合法）：连接是好的，重连只会白烧一次。
  原样回灌给模型，让它改参数。
- **传输层断了**（子进程挂掉、stdio 关闭、调用超时）：连接已经废了，
  后面每一次调用都会用同样的方式失败。不重连的话模型会一路撞到 `max_turns` 用完。

传输层故障里「调用超时」要单独看：进程可能只是慢，但 `run_coroutine_threadsafe`
的 future 超时之后那次请求仍挂在 loop 里，这条会话已经不能干净复用，
所以一样按传输层处理——只是**不允许自动重放**，因为服务端是否已经执行完全未知。
"""

from __future__ import annotations

# 退化到按文本判定：stdio 断开在不同 SDK 版本里被包成不同异常类型，
# 但错误文本长期稳定（`Connection closed` 是 MCP 协议层自己的措辞）。
_TRANSPORT_TEXT_MARKERS = (
    "connection closed",
    "connection reset",
    "broken pipe",
    "closedresourceerror",
    "brokenresourceerror",
    "endofstream",
    "end of stream",
    "server disconnected",
    "session is closed",
    "transport closed",
    "stream closed",
    "write after close",
)

_TRANSPORT_EXC_NAMES = (
    "ClosedResourceError",
    "BrokenResourceError",
    "EndOfStream",
    "BrokenPipeError",
    "ConnectionResetError",
    "IncompleteRead",
)

REASON_DISCONNECTED = "disconnected"
REASON_TIMEOUT = "timeout"
REASON_TRANSPORT = "transport_closed"

# 超时后重放会有「服务端可能已经执行过」的风险，写类工具同理
_NO_REPLAY_REASONS = frozenset({REASON_TIMEOUT})


def classify_mcp_failure(exc: BaseException) -> str | None:
    """
    判断一次 MCP 调用异常是否属于传输层故障。

    会沿 `__cause__` / `__context__` 展开：SDK 常把底层 stdio 错误包进
    `ExceptionGroup` 或自己的 `MCPError` 里。

    @param exc 调用过程中抛出的异常
    @return 传输层原因字符串；不是传输层故障则为 None
    """
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if type(current).__name__ in _TRANSPORT_EXC_NAMES:
            return REASON_TRANSPORT
        text = str(current).lower()
        if any(marker in text for marker in _TRANSPORT_TEXT_MARKERS):
            return REASON_TRANSPORT
        for nested in (current.__cause__, current.__context__):
            if nested is not None:
                stack.append(nested)
        group = getattr(current, "exceptions", None)
        if isinstance(group, (list, tuple)):
            stack.extend(item for item in group if isinstance(item, BaseException))
    return None


def replay_allowed(reason: str, *, is_write_tool: bool) -> bool:
    """
    重连成功后是否可以自动重放这次调用。

    只在「读类工具 + 明确是连接本来就断了」时重放。写类工具与超时一律不重放：
    服务端可能已经执行过一次，重放等于让模型看不见的地方多改一遍数据。

    @param reason `classify_mcp_failure` 给出的原因
    @param is_write_tool 是否写类 MCP 工具
    @return 是否允许自动重放
    """
    if is_write_tool:
        return False
    return reason not in _NO_REPLAY_REASONS


def format_unavailable(
    *,
    server: str,
    tool: str,
    reason: str,
    reconnect_tried: bool,
    timeout_sec: float = 0.0,
) -> str:
    """
    server 已判定不可用时回灌给模型的文案。

    重点是让模型**别再调这个 server**：不写清楚的话它会把剩下的 turn
    全花在同一个必然失败的调用上。

    @param server MCP Server 名
    @param tool 工具名
    @param reason 传输层原因
    @param reconnect_tried 是否已经尝试过重连
    @param timeout_sec 调用超时上限（仅 timeout 时用于提示调大配置）
    @return 工具返回文本
    """
    if reason == REASON_TIMEOUT:
        head = f"MCP {server}/{tool} 调用超时"
        if timeout_sec > 0:
            head += f"（>{timeout_sec:g}s）"
    else:
        head = f"MCP {server} 连接已断开（{reason}）"
    tail = "重连也失败" if reconnect_tried else "已停止重连"
    return (
        f"错误: {head}，{tail}，本会话不再调用该 Server。\n"
        f"[llgraph] 不要重复调用 mcp__{server}__* 的任何工具——它们会以同样的方式失败。"
        "请改用其它工具完成，或直接向用户说明该 MCP Server 不可用。"
    )


def format_tool_crashes_server(*, server: str, tool: str) -> str:
    """
    某个工具反复把 Server 打挂时的文案：只封这个工具，同 Server 其它工具照用。

    @param server MCP Server 名
    @param tool 工具名
    @return 工具返回文本
    """
    return (
        f"错误: mcp__{server}__{tool} 每次调用都会让 MCP Server {server} 断开，本会话不再执行它。\n"
        f"[llgraph] 不要重试这个工具。同一个 Server 的其它工具仍然可用，"
        "换一个工具或换一种参数形态完成，必要时向用户说明这个工具坏了。"
    )


def format_reconnected_no_replay(
    *,
    server: str,
    tool: str,
    reason: str,
    timeout_sec: float = 0.0,
) -> str:
    """
    重连成功但不自动重放时的文案。

    @param server MCP Server 名
    @param tool 工具名
    @param reason 传输层原因
    @param timeout_sec 调用超时上限
    @return 工具返回文本
    """
    if reason == REASON_TIMEOUT:
        cause = f"调用超时（>{timeout_sec:g}s）" if timeout_sec > 0 else "调用超时"
    else:
        cause = f"连接断开（{reason}）"
    return (
        f"错误: MCP {server}/{tool} {cause}；连接已重建，但这次调用**是否已在服务端生效未知**。\n"
        "[llgraph] 不要直接重试。先用只读方式确认上一次是否已生效，确认没生效再重试一次。"
    )
