# 2026-09-11 网关流式瞬时故障自动重放

选题：**稳定性**（网关失败恢复）。

`2026-09-10` 结尾点的两个方向是「性能：扩散式检索」和「商用体验：改码命中率」。
本轮没接这两条，因为翻代码时撞到一个更靠底层、也更便宜的洞：

**全仓 `llgraph/` 里一处 LLM 重试都没有**（`rg "retry|backoff" llgraph/` 只命中
`conversation_anchor.py` 一处无关文案）。出站走 `.stream()`，响应头一回来，
SDK 自带的 `max_retries` 就不再兜底：中途被网关掐断（502 / connection reset /
incomplete chunked read）会直接把整轮 ReAct 打掉——哪怕前面已经跑完十几个工具、
改了一半代码。而 `2026-09-08` 以来这条稳定性线做的会话耐久、改不动升级，
保住的都是「轮次内的状态」，没人管「这一轮还能不能活下来」。

网关空 body 更糟：`_consume_runnable_stream` 里 `response is None` 直接抛
`UserCancelledError`，被当成「用户 Stop」静默收场，用户只看到一个空回复，
`last_run.json` 里也记成 cancelled，排查时完全指错方向。

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

- `python3 -m pytest -q` → 621 passed / 4 skipped（本机基线 603 passed / 4 skipped，本轮 +18；
  本机没装 `[index,ast]` 可选依赖，skip 数与 `2026-09-10` 那台不同，不是回归）
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
- 重放循环包在 `prompt_runnable.invoke` **之外**：出站前缀与 prompt cache 度量
  （`record_dispatch_prefix`）每轮只记一次。**不要**把重放挪到 prompt 准备之前，
  那会把 `2026-09-10` 那轮的缓存前缀度量记重。
- 下一轮建议（二选一）：
  - 接着攻稳定性的**工具侧**——重放解决了出站断流，但工具异常
    （MCP server 进程挂掉、shell 子进程僵死）目前各写各的兜底，可以收敛成一层。
  - 或回到 `2026-09-10` 点名但本轮跳过的那两条：扩散式检索去重 / `search_replace` 前带真实行号。
