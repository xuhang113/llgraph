# 2026-09-07 冷启动：提示符 1.04s → 0.24s

选题：**速度（启动耗时）**。接 `2026-09-05-first-token-prelude.md` 结尾点名的那条线：
「`import llgraph.core.agent` 单独就要 2.2s，是敲下回车到出提示符之间的固定成本，
可考虑按需延迟」。上一轮把首 token 前奏从 1503ms 打到 34ms 之后，启动这一段
就成了单笔最大的墙钟，而且此前没人碰过。

先量再改。同机采样：`llgraph --help` 1174ms，交互 time-to-banner 1040ms，
其中真正干活（sandbox / watch / MCP / build_agent）只有 72ms —— **93% 是 import**。
`llgraph.core.llm` 一条链（`langchain_anthropic` + `anthropic` SDK）就 0.79s，
`langgraph.graph` 0.27s。这些只有真正建 Agent 才需要，打印一行 `--help`
不该先付这笔钱。

## 做了什么

**1. 拆掉「谁都会顺带拉起 LLM SDK」的隐式依赖**

`context/conversation_anchor.py` 顶层 `from llgraph.core.llm import create_gateway_llm`
只在 `_invoke_anchor_summary_llm` 里用，但它把 anthropic SDK 拖进了
`context_compressor → execution_log → trace_display → terminal.session` 整条展示链。
改为函数内延迟 import。同类处理：

- `context/context_compressor.py` → `langgraph.graph.message.REMOVE_ALL_MESSAGES`
- `core/checkpointer_factory.py` → `langgraph.checkpoint.memory.MemorySaver`
  （`checkpointer_kind()` 只是查文案，却要付 0.19s）
- `session/session_web_search.py`、`session_write_mode.py`、`session_sandbox.py`
  → `core.agent.rebuild_agent_preserving_memory`（启动时会调
  `resolve_initial_web_search_enabled`，等于把 Agent 全链拉在 banner 前面）

**2. `main.py` 顶层只留标准库**

重链路全部挪进函数体。argparse 要 `TraceMode`，所以把 Enum 拆到
`display/trace_mode.py`（纯 Enum，无依赖），`trace_display` 继续 re-export
同一个对象，老 import 路径不变。

**3. `load_mcp_tool_bundle` 挪到 `core/mcp_bundle.py`**

原来在 `core/tools.py`，顶层要 `langchain_core.tools` + 全部内置工具模块（0.42s），
但启动阶段跑的只是「有没有配 MCP」。新模块先读配置，没有 enabled Server 就直接返回，
不 import `core.mcp_tools`。`core.tools.load_mcp_tool_bundle` 保留 re-export。

**4. 交互模式：banner 先出来，Agent 后台建（`runtime/agent_warmup.py`）**

剩下的 import 是建 Agent 真需要的，省不掉，只能挪出关键路径。
`LazyAgent` 是个转发代理，构造时起 daemon 线程调 `build_agent`，
首轮访问 `.get_state` / `.stream` 时才阻塞等它。用户读 banner、打字的那段空档
正好把 0.5s 吃掉。

不指望「后台线程 import 更快」：GIL 下 import 是 CPU 密集，和主线程抢锁不会更快，
收益全部来自「主线程空闲等输入」这段时间。所以没有做「启动即 prewarm 线程」那种假优化。

两个必须守住的行为：

- 凭据缺失仍在 banner 前报错：延迟构建前先调一次 `get_llgraph_settings()`
  （纯 `os.getenv`，不碰 SDK），`配置错误: 缺少环境变量 ...` 的位置和文案不变
- 后台构建失败不吞不抛：`resolve()` 在主线程重跑 builder，异常照原样在调用点抛出

**5. 顺手补 `llgraph/runtime/__init__.py`**

该目录缺 `__init__.py`，`setuptools` 的 `packages.find` 会整个丢掉 —— 非 editable
安装装出来的包里没有 `runtime/shutdown.py`，退出回收会直接 ImportError。
`llgraph/prompts` 是走 package-data 的，不算。

## 改了哪些路径

- `llgraph/main.py`
- `llgraph/display/trace_mode.py`（新）、`llgraph/display/trace_display.py`
- `llgraph/runtime/agent_warmup.py`（新）、`llgraph/runtime/__init__.py`（新）
- `llgraph/core/mcp_bundle.py`（新）、`llgraph/core/tools.py`
- `llgraph/core/checkpointer_factory.py`
- `llgraph/context/conversation_anchor.py`、`llgraph/context/context_compressor.py`
- `llgraph/session/session_web_search.py`、`session_write_mode.py`、`session_sandbox.py`
- `tests/test_startup_import_budget.py`（新）、`tests/test_agent_warmup.py`（新）

## 怎么验收

墙钟（同机各 3 次，前后同一台）：

| 场景 | 改前 | 改后 |
|------|------|------|
| `llgraph --help` | 1174ms | 35ms |
| `llgraph index --help` / `search --help` | 1174ms | 75ms |
| `llgraph web --help` | 1174ms | 32ms |
| `llgraph --list-sessions` | 1190ms | 260ms |
| 交互 time-to-banner（空工作区） | 1040ms | 235ms |
| 交互 time-to-banner（`.llgraph` + 1 MCP Server） | 1040ms | 440ms |

回归测试断的是 import 图而不是墙钟（墙钟随机器抖，import 图不抖，
红的时候直接指到是谁把链拉回来的）：

- `tests/test_startup_import_budget.py`：干净子解释器里 import
  `llgraph.main` / `cli.*` / `terminal.session` / `trace_display` /
  `session_registry` / `core.mcp_bundle` / `checkpointer_factory` /
  `session_web_search`，`sys.modules` 里不许出现
  `anthropic` / `langchain_anthropic` / `langgraph`；
  反向再断 `llgraph.core.agent` 必须拉起这三个，防止用例假绿。
  另含 `trace_display.TraceMode is trace_mode.TraceMode`、
  `core.tools.load_mcp_tool_bundle` re-export、以及子包 `__init__.py` 齐全。
- `tests/test_agent_warmup.py`：转发透明、并发只建一次、后台成功被复用、
  后台失败在主线程原样抛出。

手工：`python -m pytest -q` 全绿（装了 `.[web]` 时 548 passed / 3 skipped）；
`ruff check llgraph tests` 通过；`pip install -e .` 重装后
`llgraph --help` / `python -m llgraph` / `--once` / `--thread-id` 恢复 /
`--list-sessions` / `/context` / `/tools` / `/write on` / `/model` 均正常；
banner 输出与改前逐字符一致（仅随机 thread_id 不同）；
缺凭据时仍在 banner 前打 `配置错误: 缺少环境变量 ...`。

## 未做 / 下一步不要做

- 剩下的 235ms 里 `langchain_core` 系（含 `requests` 0.11s、`langsmith.schemas` 0.07s）
  是大头，来源是 `session_file_store` 需要 `langchain_core.messages`。
  再压要把落盘层与 langchain 消息类型解耦，收益 ~0.15s，不值当，**先不要动**。
- 带 MCP Server 时多出的 ~0.2s 是真在起 stdio 子进程 + 建 StructuredTool，
  属于功能开销。要动就得改成「首轮再连 MCP」，会牵动 `/tools` 展示与工具注册时序，
  **本轮不碰**。
- 没有做「启动即开 prewarm 线程去抢 import」这类看着聪明其实无效的优化：
  GIL 下 import 是 CPU 密集，主线程同时也在 import，并发不会更快。
  收益只可能来自「把重活挪到主线程空闲等输入之后」，下一轮别再往前者试。
- 上一轮提到的 `build_agent` 里 19 个工具的 pydantic schema 生成没有再优化：
  它现在整体落在 `LazyAgent` 后台线程里，已经不在用户等待路径上，
  **不要**为它去改工具 schema 的定义方式。
- 下一轮建议换赛道：**稳定性**（工具失败恢复 / 压缩丢上下文 / 并发竞态）
  或**商用体验**（计划-多 Agent、记忆连续性）。速度这条线的低垂果实已经摘完。
