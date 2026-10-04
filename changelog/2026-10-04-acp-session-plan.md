# 2026-10-04 待办在编辑器里成了一张计划清单：ACP `plan`

选题：**排队第 2 件（编辑器里干活）**。`2026-09-25` 终端那篇末尾点名的就是这件：

> 「下一件按 `AGENTS.md` 排队走：**ACP 计划**（`session/update` 的 `plan`）——`todo_write` 的待办
> 现在只有 CLI / Web Console 看得到，编辑器里一条都不显示；难点是 `plan` 是全量快照，
> 得先决定每次 `todo_write` 之后整份发、还是只在状态真的变了时发，以及 `todo_store` 的状态
> 怎么映射成 ACP 的 `pending` / `in_progress` / `completed`。」

在此之前，编辑器里只能看到一行「执行 todo_write」，点开是一段「任务清单 1/3 完成 …」的纯文本
——模型按一张表在干活，用户看不见那张表，只能切回 CLI 敲 `/todos` 或开 Web Console。

## 做了什么

### 1. 全量快照，每次 `todo_write` 之后就发

上一篇留的那个难点（整份发还是按状态变化发）选了**整份发**。ACP 的 `plan` 本来就是整块替换，
编辑器收到就照单重画；「只在状态真的变了时发」意味着两头各留一份当前表再比对，对不上就会在
编辑器里画出一张与落盘不符的清单——而清单这东西，不准比没有更糟。

唯一挡掉的是**一模一样**的那张表：模型 `merge=true` 原样写回时（这在长 turn 里常有），
编辑器不必白刷一次。比的是发出去的那份载荷，所以「id 换了但正文与状态没变」也算同一张表——
ACP 的 `plan` 里压根没有 id，编辑器只按顺序渲染。

两处不发：

- **清空清单发空快照，但只发一次。** `todo_write(todos=[], merge=false)` 之后那张表该跟着清掉，
  所以空 `entries` 要发；连着清两次只发一条。
- **`merge=true` 不给 todos 不发。** 那只是「查一眼当前清单」，根本没落盘，报了等于说它变过。

### 2. 状态与 priority：缺的那两格宁可说清楚，不凭空造

`todo_store` 有四档状态，ACP 只有三档：

- `pending` / `in_progress` / `completed` 直通。
- **`cancelled` 归到 `completed`，正文前加 `（已取消）`。** 留成 `pending` 会让编辑器里那张表
  永远显示「还有活没干」；直接报 `completed` 又会看起来像真做完了。这一格是界面取舍，
  落盘与模型那边仍是 `cancelled`。
- `priority` 是 `PlanEntry` 的必填项，而 llgraph 的清单没有优先级这个字段。按顺序硬编一个高低
  只是凭空造数据（编辑器会当真画出高/低徽标），一律 `medium`。

### 3. 发在哪：不挂在任何一行工具调用上

这条与前三轮（diff、locations、终端块）都不一样——它们都是往某个 `toolCallId` 上挂东西，
`plan` 是**会话级**的一张表，载荷里没有 `toolCallId` 也没有 `status`。所以 Sink 这边是一条独立的
通道：`todo_write` 落盘那一刻经 `core/tool_progress.py` 的计划观察者报整张表（`notify_plan_updated`），
Sink 发一条 `plan`。那一行「执行 todo_write」照旧走自己的三段状态，文本输出也照旧回给模型。

观察者仍放在同一处 ContextVar（与起步 / 编辑 / 终端通知同一个理由：工具在 LangGraph 的线程池里跑，
全局变量会在多会话同进程时串台）。没有入口登记观察者时是空操作——CLI / Web Console 一行行为都没变。

归属不需要 `tool_call_id`：计划不属于某一次调用，这也是它比 diff / 终端块少一道「认不出就不报」的原因。

### 4. 续聊接得回那张表

清单**不在** `messages.jsonl` 里（它单独落盘，压缩与 tool 裁剪都不会丢它），所以回放历史带不出它。
不补一条的话，编辑器重启后清单是空的，而模型下一轮仍按那张表干活——两边看到的不是同一个计划。
`session/load` 的回放末尾因此按现状补一条 `plan`（空清单不发，发空快照只是白刷一次）。

排在历史之后：计划是会话当前的状态，不属于时间线上的某一条。

## 改了哪些路径

- `llgraph/core/tool_progress.py`（新增 `PlanItem` / 计划观察者 / `notify_plan_updated`）
- `llgraph/core/todo_tools.py`（落盘之后报整张表）
- `llgraph/editor/acp/updates.py`（`plan_entries` / `session_plan`、四档 → 三档、`priority`、条数上限）
- `llgraph/editor/acp/sink.py`（`todo_plan`：全量发 + 挡掉一模一样的那张表）
- `llgraph/editor/acp/turn.py`（登记计划观察者）
- `llgraph/editor/acp/replay.py`（`plan_update`：续聊末尾补当前清单）
- `tests/test_acp_plan.py`（新，17 例）、`tests/test_acp_end_to_end.py`（+1 例）
- `README.md`、`docs/模块说明.md`、`docs/项目结构.md`、`AGENTS.md`、`.cursor/rules/cloud-agent.mdc`

## 怎么验收

- `python3 -m pytest tests -q`（除 `test_prompt_cache_breakpoints.py`）→ **1142 passed, 10 skipped**
  （本轮前 1124；新增 18 例）；`ruff check llgraph tests` 通过。
  `tests/test_prompt_cache_breakpoints.py` 单跑仍是 **11 failed**，与本轮无关且与前几轮同因
  （本机装的 `langchain-anthropic` 里 `_format_messages()` 多了必填关键字参数 `model`）：
  把本轮改动 stash 掉重跑，同样 11 failed。
  skip 全是本机没装的可选依赖（`fastapi` / `lancedb` / `langchain-openai` / `langchain-ollama` /
  `langchain-google-genai`）
- 载荷：四档状态按上面的表映射、`cancelled` 带 `（已取消）` 前缀且报 completed、
  `priority` 一律 `medium`、认不出的状态按 `pending` 算（而不是丢掉那一条）、
  正文里的换行折成一行、脏条目（None / 字符串 / 空正文 / 缺正文）丢掉、超过 20 条截断、
  清空发的是空 `entries` 而不是什么都不发
- 通知来源：作用域随 with 块结束、空表也报（清空是一次变化）、作用域外报不抛、
  观察者自己抛异常不外溢；`todo_write` 报的是**整张表**（不是本次提交的那几条）且与落盘的一致；
  `merge=true` 不给 todos 不报；没登记观察者时 `todo_write` 的返回与落盘照旧
- Sink：每次变化发一条、一模一样的表不重发（含「id 变了但正文与状态没变」）、
  清空只发一次、载荷里没有 `toolCallId` / `status`，那一行工具调用照旧走三段状态
- 续聊：有清单时补一条 `plan`、没清单时一条都不发、`load_session_updates` 把它排在历史之后
- 端到端（stub server 说 Anthropic 协议，**真跑 ReAct**）：模型调 `todo_write` 那轮收到
  一条 `plan`（三条 entries，状态 completed / in_progress / pending，priority 全 medium），
  它排在工具收尾那条**之前**，工具那一行仍是 `pending` → `in_progress` → `completed`；
  同一个会话再走 `session/load`，最后一条就是那张表
- `pip install -e .` 后 `llgraph --help`、`llgraph acp --help`、`python3 -m llgraph --help` 正常

## 未做 / 下一步不要做

- **不要**改成按状态变化发增量。ACP 的 `plan` 是全量快照，没有「更新第 N 条」这种方法；
  自己攒增量得在两头各维护一份当前表，对不上就会画出一张与落盘不符的清单。
- **不要**给 `PlanEntry` 造优先级。llgraph 的清单没有这个字段，按顺序或按状态硬编一个高低
  会让编辑器画出用户没设过的高/低徽标。真要做得先在 `todo_write` 的参数里加 `priority`，
  并想清楚它跟「最多 1 条 in_progress」这个约束怎么共存。
- **不要**把 `cancelled` 改回报 `pending`。那会让编辑器里那张表永远显示「还有活没干」；
  落盘与模型那边仍是 `cancelled`，这一格只是界面取舍。
- **不要**把计划挂到那一行 `todo_write` 的 `content` 上（也不要因为有了 `plan` 就把工具的文本输出
  从**模型**那边删掉）。前者会让同一张表在编辑器里出现两份，后者会让模型不知道自己刚写了什么。
- **不要**给 `/todos clear`（CLI 的那条斜杠命令）补通知。它只在 CLI 里走，ACP 会话压根到不了那段代码；
  真要做得先解决「编辑器里的斜杠命令」这件事（见下）。
- 一轮开始时不重发清单：编辑器拿着上一轮那张表，`plan` 又是全量替换，重发只是白刷一次。
  跨轮的第一条确实可能发一份与上一轮末尾一样的快照（去重状态按轮重置），全量替换是幂等的，
  不值得为此把去重状态搬到会话层。
- 下一件按 `AGENTS.md` 排队走：**ACP 会话模式**（`session/set_mode` + `currentModeUpdate`
  + `initialize` 里报 `modes`）——「只读 / 每次问 / 免确认」现在是 `llgraph acp` 的启动参数，
  在编辑器里想换得退出 Agent 重开；难点是 `allow_write` 决定了工具集与 Runtime
  （`RUNTIME_MANAGER.get(workspace, allow_write=)`），换模式得在**轮与轮之间**生效而不是中途，
  还要决定切到只读时那些已经「本会话都允许」的授权记忆怎么办。
  之后一件是**编辑器里的斜杠命令**（`availableCommands`）：llgraph 的 `/model`、`/compact`、`/todos`
  现在只有 CLI 认，ACP 的 prompt 根本不过那层命令处理。VS Code 扩展与终端 TUI 仍后置。
