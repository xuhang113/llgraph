# 2026-09-22 工具跑之前就看得到：ACP `tool_call` 三段状态

选题：**排队第 2 件（编辑器里干活）**。`2026-09-21` 缓冲区那篇末尾点名的就是这件：

> 「下一件按 `AGENTS.md` 排队走：**`tool_call` 的中间态**——现在 `sink.py` 只在工具跑完
> 才推一条完成态 `tool_call`，编辑器里一个长命令（跑测试）期间看不到任何动静；
> 要补 `pending` → `in_progress` → `completed` 三段更新，难点是 trace 那边现在只在步末给事件，
> 得先有「步开始」这个钩子，且同一个 toolCallId 要在三段之间稳住。」

在此之前，编辑器里一条 `run_terminal_cmd` 跑测试的几分钟内屏幕上什么都不会变，
跑完才「唰」地出现一整条带输出的记录——用户分不清是模型在想、工具在跑，还是进程卡死了。

## 做了什么

trace 的步骤记录**天生只在工具跑完后才存在**（那一刻才有耗时与输出），这是它的设计，
不该为了界面把它改成「先登记再回填」。所以这轮是**另接两条来源**，完成态仍走原路：

| 状态 | 来源 | 时机 |
|------|------|------|
| `pending` | `TurnTracePrinter.on_agent_update` → `sink.tool_calls_planned()` | 模型刚决定要调哪些工具 |
| `in_progress` | `ToolNode._run_one` → `tool_progress` 观察者 → `sink.tool_started()` | 这一次调用真正开跑 |
| `completed` | 原有的 `step_added`（改发 `tool_call_update`） | 工具返回、步骤登记时 |

两段的间隔不是凑数的：中间隔着上下文维护、写串行化、以及排在别人后面——
并行三个工具时，先跑的那条已经 `in_progress`，另外两条还老实待在 `pending`。

### 1. `core/tool_progress.py`：单次调用的「起步」通知

链路里唯一「一次工具调用从头包到尾」的位置是 `tool_invoke_timing.wrap_tool_node_with_timing`
给 `_run_one` / `_arun_one` 加的那层计时包装，所以起步通知就搭在计时起点上——
同一个括号里，不用再叠一层包装，也不可能与耗时记录错位。

观察者放在 ContextVar 上，与 `permissions/approval.py` 的闸门、`core/editor_fs.py` 的文件来源
同一个形状：工具在 LangGraph 的线程池里跑，全局变量会在多会话同进程时串台。
**没登记观察者时一律空操作**，CLI / Web Console 一行行为都没变；
观察者自己抛异常一律吞掉——它只是给界面加进度，不能把工具执行带崩。

### 2. 三段靠模型给的 `tool_call.id` 串成一行

`TraceStepRecord` 多一个 `tool_call_id` 字段（`on_tools_update` 从 ToolMessage 上取），
Sink 据此把「跑完的这条」认回「之前报过 pending 的那条」。三个决定：

- **不靠标题配对。** 同一轮里 `read_file(a.txt)` 完全可能被调两次，标题一模一样。
- **前缀按轮递增**（`t1_call_1`、`t2_call_1`…）。Sink 是每轮新建的，计数器也从头开始，
  不加前缀的话第二轮的 `call_1` 会撞上第一轮那条已收尾的调用，编辑器按 `toolCallId` 认行，
  一撞就把新调用画到旧记录上。前缀由 `server.py` 按会话的 `turn_seq` 给。
- **收尾那条不重发 `title` / `kind`。** 工具节点的输出里没有调用参数，照它重算标题会从
  `执行 read_file(hello.txt)` 退化成 `执行 read_file`——编辑器里那行字跑完反而变模糊了。
  `tool_call_update` 只报状态与输出，标题沿用 pending 那条。

### 3. 三处「不报」比「报了」重要

- **`spawn_subagent` 不报 pending。** 它的完成态由 `emit_explore_trace_step` 以 explore 步骤登记，
  `on_tools_update` 那边本来就跳过它；报了 pending 就再没人给它收尾，界面上永远转圈。
- **没报过 pending 的调用不发 `in_progress`。** 宁可少一段状态，也不要冒出一条不会收尾的记录。
- **verbose（`trace all`）下一条都不报。** 那个模式按行打印工具输出、不按工具登记步骤，
  同理没人收尾。

真收不了尾的情况（本轮被取消、模型那边中途抛错）由 `run_acp_turn` 收场时兜住：
把还开着的那几条标 `failed`。连接已经断了也不再抛——收场路径上再抛会盖掉真正的错误。

## 改了哪些路径

- `llgraph/core/tool_progress.py`（新）
- `llgraph/core/tool_invoke_timing.py`（计时包装里加起步通知）
- `llgraph/display/trace_display.py`（`TraceStepRecord.tool_call_id`、`_notify_tool_calls_planned`）
- `llgraph/editor/acp/sink.py`（三段状态与 id 映射）、`updates.py`（`tool_call_pending` / `tool_call_status` / `as_update`）
- `llgraph/editor/acp/turn.py`（登记观察者、收场标 failed、接前缀）、`server.py`（按会话的 `turn_seq`）
- `tests/test_acp_tool_progress.py`（新，19 例）、`tests/test_acp_server.py`（+1 例）、`tests/test_acp_end_to_end.py`（+1 例）
- `README.md`、`docs/模块说明.md`、`docs/项目结构.md`、`AGENTS.md`、`.cursor/rules/cloud-agent.mdc`

## 怎么验收

- `python3 -m pytest tests -q` → **1051 passed, 10 skipped**（本轮前 1030；新增 21 例、改 2 例断言。
  skip 全是本机没装的可选依赖：`fastapi` / `lancedb` / `langchain-openai` / `langchain-ollama` / `langchain-google-genai`）；
  `ruff check llgraph tests` 通过
- 观察者语义：没登记时 `notify_tool_started` 空操作、作用域随 with 块结束、
  观察者抛异常不外溢、空 id 不通知；计时包装在**工具真正跑之前**通知（同步与 async 两条路都测），
  且耗时照旧记得下
- Sink：pending → in_progress → completed 是同一个 `toolCallId`，且新建一条、更新两条；
  并行工具按 `tool_call_id` 各归各位（完成顺序与规划顺序相反也不串）；
  重复的规划通知只发一次；没报过 pending 的 id 不发 in_progress；已完成的调用不会被迟到的起步通知推回进行中；
  非工具步骤（模型决策 / 回复）不占号；explore 步骤仍按新建发；
  收场时没收尾的标 `failed`（只标一次），连接断了也不抛
- trace 侧：`on_agent_update` 给出 id / 工具名 / 带参数的标题，`spawn_subagent` 与无 id 的调用被跳过，
  纯回复那轮不通知，verbose 模式不通知；`on_tools_update` 把 ToolMessage 的 `tool_call_id` 记进步骤
- 协议层：同一会话连提两轮，第二轮的 `tool_call_prefix` 是 `t2_`（跨轮不撞号）
- 端到端（stub server 说 Anthropic 协议，**真跑 ReAct**）：一轮读文件收到的工具更新正好是
  `pending` → `in_progress` → `completed` 三条、同一个 id、pending 那条就带 `执行 read_file(hello.txt)` 与 `kind=read`，
  三条都排在正文之前
- 真子进程（像编辑器那样 `python -m llgraph acp -C <ws>`，stdin/stdout 交互）：
  收到 `tool_call t1_call_1 pending 执行 read_file(hello.txt)` → `tool_call_update in_progress`
  → `tool_call_update completed`（带文件内容）→ 正文，`stopReason: end_turn`
- `pip install -e .` 后 `llgraph --help`、`llgraph acp --help`、`python3 -m llgraph --help` 正常

## 未做 / 下一步不要做

- **不要**把 trace 的步骤改成「工具开始时先登记、跑完再回填」。步骤记录是终端 / Web / ACP
  三个入口共用的，它的语义是「已完成的一步」（耗时、输出、token 都在里面）；
  改成可变的半成品，终端折叠行与 Web 落盘（`live_web_trace.json` 按 `step_id` upsert）都要跟着动。
  中间态是**入口**的事，本轮就是按这个分工做的。
- **不要**把起步通知从计时包装里挪出去单独包一层 `_run_one`。两层包装顺序一乱，
  「开始」就可能晚于「完成」，而且计时与通知本来就该用同一个括号。
- **不要**给 `pending` 载荷塞 `rawInput`。看着更全，但 `write_file` 的参数里是整份文件正文，
  每次规划都往编辑器推一遍，长文件直接把通道打满。
- **不要**在 `in_progress` 里报「已经跑了多久」。ACP 没有这个字段，硬塞进标题就得按秒重发更新，
  一条长命令能刷出几百条 `session/update`。
- 完成态仍一律是 `completed`：工具返回像失败（`_tool_output_looks_like_error` 认得出来）时
  没有报 `failed`。要做得把这个判定结果一起记进步骤记录，顺带决定 Web 侧要不要也跟着变色。
  **下一轮就是这件**，连着 `locations`（`tool_call` 带上受影响文件的绝对路径，编辑器才能点开跳转）一起。
- 授权弹窗那条仍用独立的 `perm_*` id，没有和本轮的 `t{n}_call_{m}` 合成一条：
  弹窗在 `write_file` 落盘那一刻发出，比工具级的 id 更细（一次调用可能问多次），
  真要合得先决定「一次调用多次授权」在编辑器里画成几行。
- `pending` 的标题用的是 trace 那套参数摘要（`_short_tool_target`），MCP 工具的参数名不在它的
  优先表里时只剩工具名。要补先想清楚 MCP 参数哪个字段算「目标」。
