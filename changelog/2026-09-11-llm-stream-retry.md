# 2026-09-11 网关流式瞬时故障自动重放

选题：**稳定性**（工具/网关失败恢复）。

上一条 changelog 留的方向是「在已有工具层方向里接着做深一件」。翻代码发现一个更靠底层的洞：
全仓 `llgraph/` 里一处 LLM 重试都没有。出站走 `.stream()`，响应头一回来 SDK 自带的
`max_retries` 就不再兜底，中途被网关掐断（502 / connection reset / incomplete chunked read）
会直接把整轮 ReAct 打掉——哪怕前面已经跑完十几个工具、改了一半代码。
网关空 body 更糟：以前被当成「用户 Stop」静默收场，用户只看到空回复。

## 做了什么

- 新增 `llgraph/core/llm_retry.py`：瞬时故障分类 + 退避节奏
  - 按异常类名、HTTP 状态码、消息文本三路判定，不 import anthropic / httpx（不破坏冷启动 import 预算）
  - 会展开 `__cause__` / `__context__`：SDK 常把 400 包在看着像网络错的壳里，内层致命就不重放
  - 上下文超限、鉴权、400 一律不重放（重放只会再烧一次同样的 prompt）
  - 指数退避 + 抖动，上限可配
- `llgraph/core/react_invoke.py`：`_consume_runnable_stream` 拆成「单次流式 `_stream_once`」+「外层重放循环」
  - **重放门槛**：本次尝试已经流出可见正文就不再重放（否则用户看到半截重复）；
    thinking 与半截 tool_call 都可以丢掉重来——它们没进 state，终端也没当答复渲染
  - 网关空 body 从「假装用户 Stop」改成可重放的 `EmptyGatewayStreamError`；
    重放耗尽后仍按原来的 `UserCancelledError` 收场，不把内部异常泄给 CLI
  - 退避期间保持 Stop 可响应（`_sleep_with_cancel` 按 50ms 轮询 cancel）
  - 重放事件写 `run_log.jsonl`：`llm_stream_retry` / `llm_stream_retry_ok` / `llm_stream_retry_give_up`
- 可配（`.llgraph/agent.json`，都有默认值，不配也能跑）：

```json
{ "llm": { "stream_retry": { "max_attempts": 3, "base_delay_sec": 1.0, "max_delay_sec": 20.0 } } }
```

## 改了哪些路径

- `llgraph/core/llm_retry.py`（新增）
- `llgraph/core/react_invoke.py`
- `tests/test_llm_stream_retry.py`（新增，18 例）

## 怎么验收

- `python3 -m pytest -q` → 621 passed / 4 skipped（基线 603，本轮 +18）
- `python3 -m ruff check llgraph tests` 全绿
- `pip install -e .` 后 `llgraph --help` 正常；`test_startup_import_budget.py` 仍绿（新模块没把 SDK 拉进启动早期路径）
- 关键用例语义：
  - 502 / connection reset / 空 body → 重放并成功返回
  - 已流出可见正文后断流 → 不重放，原样抛出
  - 半截 tool_call 后断流 → 重放
  - 400 / 上下文超限 / 内层 400 包在网络错壳里 → 不重放
  - 退避窗口里按 Stop → 立刻 `UserCancelledError`，不等满退避

## 未做 / 下一步不要做

- 没动异步路径 `ainvoke_agent_runnable_cancellable`：仓库里没人调 `agent.ainvoke/astream`，
  真要用再补，不要现在为对称性而改。
- 没做「重放时降级换模型 / 换网关」：那是另一条决策线，别和瞬时重试混在一起。
- 没把重试提示搬进对话区（仍走 `ops_notice`，需 `LLGRAPH_VERBOSE_CONTEXT=1`）。
  真要做请先给 printer 加一个通用 notice 钩子，不要在 `react_invoke` 里直接 print。
- 下一轮建议：接着攻稳定性里的「工具侧」——重放解决了出站断流，
  但工具异常（MCP server 挂掉、shell 子进程僵死）目前还是各写各的兜底，可以收敛成一层。
