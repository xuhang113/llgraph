# 2026-09-12 MCP 工具侧：先修「跑成功被报成失败」，再补「挂掉之后怎么活」

选题：**稳定性（工具失败恢复 / 死循环）**。接 `2026-09-11` 结尾点名的第一条：

> 「重放解决了出站断流，但工具异常（MCP server 进程挂掉、shell 子进程僵死）
> 目前各写各的兜底，可以收敛成一层。」

本轮只做 **MCP** 这一半，**没有动 shell**，理由见文末。

## 为什么是这个切口

先量再改。写了个真 stdio MCP server（两个工具：`echo` 正常返回，`boom` 调用时
`os._exit` 自杀），用 `McpToolRegistry` 真连真调。改前：

| 调用 | 改前 | 改后 |
|---|---|---|
| `echo` #1（正常） | `MCP 调用失败 (probe/echo): 'CallToolResult' object has no attribute 'isError'` | `echo:hi` |
| `boom`（打挂进程） | `MCP 调用失败: Connection closed` | 封停这一个工具，明确要模型别再调它 |
| `echo` #2~#5 | 4 次全是 `Connection closed` | 4 次全部正常返回 |

两个独立的洞，第一个比第二个严重得多。

### 洞 1：SDK 2.x 把字段改了名，于是「已经跑成功」被报成「调用失败」

`pyproject` 写的是 `mcp>=1.0.0`，今天 `pip install -e ".[mcp]"` 装到的是 2.x。
SDK 2.0 把 `Tool.inputSchema` → `input_schema`、`CallToolResult.isError` →
`is_error`、`structuredContent` → `structured_content`。这些都是 pydantic 模型：
**读不存在的字段抛 `AttributeError`，不是返回 `None`**。于是：

- `result.isError` 抛异常 → 被 `call_tool_sync` 的 `except Exception` 接住 →
  变成一句「MCP 调用失败」。**服务端其实已经执行完了。** 模型看到失败就重试，
  写类 MCP（`allow_write_tools`）下等于把一次已经生效的写又做了一遍。
- `hasattr(mcp_tool, "inputSchema")` 为假 → `input_schema` 取成 `{}` →
  所有 MCP 工具退化成 `arguments_json` 文本入参，模型丢掉结构化字段，
  工具描述里也不再带 JSON Schema。

也就是说：**2.x 下 MCP 整条链路是坏的**，而且坏的方式是「报错」而不是「崩」，
所以一直没人发现。

### 洞 2：进程挂了之后没人管

`_McpServerRuntime` 连上就 `run_forever`，没有任何健康状态。子进程一挂，
本会话余下每一次调用都以同样的方式失败，既不重连，也不告诉模型别再试。
`max_turns` 默认 100，模型会一路撞到「Sorry, need more steps」。
1.x SDK 下更糟：断开后请求没人应答，每次还要等满 `defaults.timeout_sec`（默认 60s）。

顺带两个小洞：`future.result(timeout=)` 超时后不 `cancel()`，协程一直挂在 loop 里；
等待期间按 Stop 要等满 `timeout_sec` 才有反应。

## 做了什么

### 1. `llgraph/core/mcp_compat.py`（新）：按两套命名取值

`input_schema` / `inputSchema`、`is_error` / `isError`、
`structured_content` / `structuredContent` 都试一遍，取不到才落保守默认
（schema 退 `{}`、is_error 退 `False`）。内容块为空时退回结构化返回，
避免只给结构化输出的工具被渲染成空字符串。

### 2. `llgraph/core/mcp_health.py`（新）：分类，然后三条规则

**先分类**，因为两类失败的处置完全相反：

- **工具自己报错**（`is_error=True`、SQL 语法错、参数不合法）：连接是好的，
  重连只会白烧一次。原样回灌，让模型改参数。
- **传输层断了**（`Connection closed` / `ClosedResourceError` / 调用超时）：
  连接已经废了，后面每一次调用都会以同样方式失败。

判定按异常类名 + 错误文本两路，并沿 `__cause__` / `__context__` /
`ExceptionGroup.exceptions` 展开——SDK 把 stdio 错误包在 `MCPError` 和
anyio 的 TaskGroup 异常组里。

**然后三条规则**（都在生产链路上，不是纯函数摆设）：

1. **重连后只自动重放读类工具。** 写类工具（复用 `permissions/mcp.is_write_mcp_tool`）
   与超时一律不重放：服务端可能已经执行过一次，重放等于在模型看不见的地方
   把数据改两遍。这两种情况改成回灌「连接已重建，但**是否已在服务端生效未知**；
   不要直接重试，先只读确认」。
2. **终止性由重连预算兜住。** `defaults.max_reconnects`（默认 3，`0`=关闭，上限 10）
   用完就判定 Server 不可用，此后**纯内存快速失败**，文案明确要模型别再调
   `mcp__<server>__*` 的任何工具。没有这一条，「进程一起来就挂」会变成无限重启。
3. **一个坏工具只封它自己。** 某个工具连着打挂两代连接才封这个工具，
   同 Server 其它工具照用（`boom` 把 server 打挂，`echo` 不该跟着陪葬）。

### 3. 等待期间 Stop 可响应

`_wait_future` 按 200ms 分片轮询，命中 cancel 就 `future.cancel()` 并返回
标准停止文案；超时同样 `cancel()`，不再把协程留在 loop 里。
刻意用 `futures.wait` 而不是 `future.result(timeout=)`：后者在 3.11+ 抛的就是
内建 `TimeoutError`，和工具自己抛的超时撞在一起分不开。

### 一个自己踩的坑，值得记下来

并发工具调度下，**同一次断开会被四五个线程同时撞到**。第一版按「失败次数」
数打挂，4 个线程撞一次断开就把计数顶到封停线上——一个完全无辜的工具被永久封掉，
而且 4 个线程各自还想重启一次进程。改成**按连接代号（generation）去重**：
打挂计数和重连预算都只认「第几代连接死了」，一次断开只算一次。

这个坑是并发回归用例第一次跑就照出来的。但接下来还有第二层：
加了状态锁之后，让 4 个线程「自然抢」已经抢不出来了——锁会把后来者挡到重连之后，
去重逻辑一次都跑不到。**变异验证里把去重退回按次数计数，21 个用例仍然全绿。**
最后把假 session 改成「卡住调用直到 4 个线程都进来，再一起报断开」，
确定性地造出这个场面，两处去重的退回才都会挂。

## 改了哪些路径

- `llgraph/core/mcp_compat.py`（新）、`llgraph/core/mcp_health.py`（新）
- `llgraph/core/mcp_tools.py`（字段兼容接线 + 重连 / 封停状态机）
- `llgraph/config/mcp_config.py`（`defaults.max_reconnects`）
- `tests/test_mcp_compat.py`（新，6 例）、`tests/test_mcp_recovery.py`（新，21 例）
- `docs/项目结构.md`、`docs/模块说明.md`

配置（都有默认值，不配也能跑）：

```json
{ "defaults": { "max_reconnects": 3 } }
```

## 怎么验收

回归测试断的是**语义边界**，不是文案。刻意不依赖可选依赖 `mcp`（CI 只装 `[web]`）：
运行时的连接步骤用假 session 顶掉，重连 / 封停走的仍是生产代码。

- 字段兼容：1.x / 2.x 两套命名都要认；**成功不许被报成失败**；缺字段退保守默认；
  只有结构化返回时不许渲染成空
- 分类：`Connection closed` / 异常类名 / 藏在 `__cause__` 与 `ExceptionGroup`
  里的都算传输层；SQL 语法错、表不存在不算（不许触发重连）
- 重放：读类 + 明确断开才重放；写类不重放；超时不重放
- 终止性：每次调用都打挂进程的 server，12 次调用后重连次数 ≤ 预算，
  收尾一定落在明确的「别再调了」上
- 快速失败：判定不可用后连打 20 次，总耗时 < 1s（改前是每次等满 `timeout_sec`）
- 坏工具封停后不再执行、不再烧预算；同 Server 其它工具仍可用
- 重连本身失败 → 判定 Server 不可用
- 并发：4 线程撞同一次断开 → 只重启 1 次、打挂计数只加 1、4 个线程都拿到正常结果
- Stop：等待期间按 Stop，`timeout_sec=30` 也在 5s 内返回

**变异验证**（8 处逐个退回，对应用例必挂）：`render_call_result` 只读 `isError`、
`tool_input_schema` 只读 `inputSchema`、打挂计数不按代号去重、重连不看代号、
写类工具也重放、超时也重放、去掉不可用后的快速失败、分类不展开 `__cause__`。

手工：

- 真 stdio MCP server 端到端（上面那张表），`boom` 之后 `echo` 立刻恢复
- `python3 -m pytest -q` → **651 passed, 3 skipped**（本轮前 624 + 新增 27；
  本机没装 `[index]` 可选依赖，3 skip 与基线一致）
- `python3 -m ruff check llgraph tests` → All checks passed
- `pip install -e .` 后 `llgraph --help` 32~33ms（与 `2026-09-10` 的 32~34ms 同档，
  新模块只在 MCP 调用链路上）；`llgraph --list-sessions`、`python3 -m llgraph --help` 正常；
  `test_startup_import_budget` 仍绿

## 未做 / 下一步不要做

- **没有动 shell 那一半**（`shell_tools` / `sandbox.exec`）。翻过了：shell 侧已经有
  硬超时、`max_jobs` 上限、`cancel_check`、相同命令去重，形态和 MCP 完全不同
  （MCP 的问题是**长连接会坏**，shell 是一次一个进程）。硬收敛成「一层」只会做出
  一个两边都不贴的抽象。要做请先量出 shell 真实的僵死形态（子进程 fork 出孙进程
  后不退、pipe 写满阻塞），那是独立一轮。
- **不要**把 `pyproject` 的 `mcp>=1.0.0` 改成 `mcp<2` 来「修」洞 1。锁在 1.x 等于
  放弃 2.x 的全部修复，而且 `mcp_compat` 已经两边都认；真要收紧版本，
  该收的是下界不是上界。
- **不要**把重连做成「每次失败都重连」。预算存在的意义就是拦住「进程一起来就挂」
  的无限重启；去掉上限，模型会拿整轮 `max_turns` 陪一个起不来的 server。
- **不要**把「写类工具不自动重放」改成「都重放」，哪怕看起来命中率更高。
  `is_write_mcp_tool` 是关键词判定、偏宽松（描述里出现 `update` 就算写），
  这个方向的误判是安全的：宁可多问一次，不要悄悄改两遍数据。
- **不要**把打挂计数从「按连接代号」改回「按失败次数」，也**不要**把并发用例
  改回「让线程自然抢」。上面那个坑的两层都在这两句上。
- **不要**在重连后重建 LangChain 工具列表。工具名不变，`StructuredTool` 闭包持的是
  runtime 对象本身，换掉内部 session 就够了。真要处理「重连后工具表变了」，
  得先想清楚它和 prompt cache 前缀的关系（工具定义在系统提示里，改了就击穿缓存）。
- **不要**把重连 / 封停事件写进对话区。现在走 `logger.warning` + `ops_notice`
  （需 `LLGRAPH_VERBOSE_CONTEXT=1`），和 `2026-09-11` 的重放提示一致。
- 下一轮建议（二选一）：
  - **性能**里还没碰过的「无效工具调用」——同一问里 grep 同一 pattern 只换 `path`
    的扩散式检索（`2026-09-10` 点名两轮了，仍是真切口）。
  - 或**商用体验**的「改码命中率」：`search_replace` 前是否该强制带上目标段的
    真实行号（`2026-09-09` / `2026-09-10` 连续点名）。注意 `edit_apply` 已经有
    exact / newline / trailing_ws / indent / whitespace / unique-fuzzy 六层容错，
    先量清楚「剩下的失败里有多少是行号能救的」再动手，不要直接加参数。
