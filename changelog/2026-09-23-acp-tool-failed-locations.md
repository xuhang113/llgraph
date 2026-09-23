# 2026-09-23 失败就标失败、改哪个文件就能点开：ACP 工具 `failed` 与 `locations`

选题：**排队第 2 件（编辑器里干活）**。`2026-09-22` 三段状态那篇末尾点名的就是这件：

> 「完成态仍一律是 `completed`：工具返回像失败（`_tool_output_looks_like_error` 认得出来）时
> 没有报 `failed`。**下一轮就是这件**，连着 `locations`（`tool_call` 带上受影响文件的绝对路径，
> 编辑器才能点开跳转）一起。」

在此之前，编辑器里一次 `search_replace` 没匹配上 `old_string`、一次 `write_file` 参数缺了 `content`，
    10|画出来和成功那次一模一样：都是一行灰字加一段输出，用户得自己点开读才知道这一刀没落地。
而那一行写着「执行 write_file(src/a.py)」，却点不开 `src/a.py`——路径只是标题里的一段文字。

## 做了什么

### 1. 失败态：判定在 trace，报状态在入口

工具**不抛异常也可能是失败的**：参数校验错（`Field required`）、`old_string` 没匹配上、
命令返回一段报错文本。LangGraph 这边它们都是正常返回的 ToolMessage，
所以「这次算不算失败」只能看返回内容。

    20|这个判定 trace 早就有了（`_tool_output_looks_like_error`，终端 verbose 下据此截断预览），
本轮**没有再判一次**，而是把它的结果记进 `TraceStepRecord.tool_failed`，
`tool_call_from_step` 照它报 `failed` / `completed`：

- **不在 ACP 那层重新认错**。同一段输出在终端里算失败、在编辑器里算成功，没法解释；
  以后要补 marker 也只有一处。
- **判定口径一个字没动**。marker 表（`错误:` / `缺少必填` / `validation error` /
  `field required` / `未找到 old_string`）是终端 trace 与 ACP 共用的，动它会顺带改终端的预览行数，
  那是另一件事（见「不要做」）。
    30|- **失败也照发输出**。`failed` 那条仍带 `content`，点开就是失败原因；
  只报个状态等于把人赶去看终端。

`tool_failed` 是步骤记录上的新字段，Web 落盘（`live_web_trace.json`）跟着多一个键、
现有前端不读它就当没有——本轮没给 Web 变色，那是 Console 的取舍，不该在 ACP 这轮顺手改。

### 2. locations：路径从参数里取，绝对化在入口做

ACP 的 `ToolCallLocation` 只认绝对路径，而工具参数里几乎都是相对工作区的路径。
分两半：

    40|| 这一半 | 在哪 | 为什么在那 |
|--------|------|------------|
| 哪个参数是「受影响文件」 | `trace_display._tool_call_paths` | 工具参数语义本来就归 trace 那套摘要管（`_short_tool_target` 就在隔壁） |
| 相对路径 → 绝对路径 | `updates.tool_call_locations` + Sink | 只有入口知道工作区根；trace 是三个入口共用的，不该认某一个的根 |

规划通知（`tool_calls_planned`）里多带一个 `paths`，Sink 补齐后放进 `pending` 载荷。
收尾那条不重发 `locations`，与标题 / kind 同一个道理：编辑器认 `toolCallId`，第一条报过就够了。

三个「不报」：

- **搜索类工具的 `path` 不算受影响文件。** `grep_files` / `glob_files` / `list_directory`
    50|  的 `path` 是扫描范围，多半就是 `.`，指过去只会打开工作区根目录。
  只报 `read_file` / `read_files` / `write_file` / `append_file` / `search_replace` 的路径参数。
- **`path="."` 不报。** 同上，那是整个工作区，不是一份文件。
- **补不出绝对路径就不报。** 没有工作区根时相对路径直接丢掉——报一条编辑器打不开的路径，
  用户点下去得到一个报错，不如那一行本来就点不动。

同一份文件在 `read_files` 里报两次会在编辑器上挂两条一样的跳转，所以去重；
一次调用最多报 8 个文件（`read_files` 可以一口气传几十个）。

## 改了哪些路径
    60|
- `llgraph/display/trace_display.py`（`TraceStepRecord.tool_failed`、`_tool_call_paths`、规划通知带 `paths`）
- `llgraph/editor/acp/updates.py`（`tool_call_locations`、`tool_call_pending(locations=)`、收尾态 `failed`）
- `llgraph/editor/acp/sink.py`（接工作区根，pending 带 locations）、`turn.py`（把工作区根给 Sink）
- `tests/test_acp_tool_result_state.py`（新，13 例）、`tests/test_acp_end_to_end.py`（+1 例、+1 条断言）
- `README.md`、`docs/模块说明.md`、`docs/项目结构.md`、`AGENTS.md`、`.cursor/rules/cloud-agent.mdc`

## 怎么验收

- `python3 -m pytest tests -q` → **1054 passed, 10 skipped, 11 failed**；
    70|  `ruff check llgraph tests` 通过。
  11 个失败全在 `tests/test_prompt_cache_breakpoints.py`，与本轮无关：本机装的
  `langchain-anthropic` 版本里 `_format_messages()` 多了一个必填关键字参数 `model`，
  `git stash` 掉本轮改动后同样是这 11 个（本轮前 1051 passed 的那台机器版本更旧）。
  skip 全是本机没装的可选依赖（`fastapi` / `lancedb` / `langchain-openai` / `langchain-ollama` /
  `langchain-google-genai`）
- 失败态：像报错的工具输出在步骤上记成 `tool_failed=True`，正常输出记成 `False`；
  载荷侧新建与更新两条路都报 `failed` 且仍带输出；没这个标记的照旧 `completed`
- locations：相对路径按工作区根补成绝对、绝对路径原样、重复路径去重；
  没有工作区根时相对路径被丢掉、绝对路径仍报；`None` / 空串 / 非字符串都不炸；
    80|  一个都补不出来时 `pending` 载荷里没有 `locations` 这个键
- trace 侧：`write_file` 报 `path`、`read_files` 报去重后的 `paths`、
  `grep_files` 与 `run_shell_command` 一条都不报、`path="."` 不报
- Sink：带工作区根时 `pending` 上是绝对路径；失败那条按
  `pending` → `in_progress` → `failed` 收在同一个 `toolCallId` 上，
  且收场的 `abandon_open_tool_calls` 不会再给它补一条
- 端到端（stub server 说 Anthropic 协议，**真跑 ReAct**）：读文件那轮 `pending` 带
  `locations=[{"path": "<ws>/hello.txt"}]`；模型少传 `path` 那轮三段是
  `pending` → `in_progress` → **`failed`**、输出里有校验原因、没有 `locations`
- `pip install -e .` 后 `llgraph --help`、`llgraph acp --help`、`python3 -m llgraph --help` 正常
    90|
## 未做 / 下一步不要做

- **不要**把 `_tool_output_looks_like_error` 的 marker 表扩大（比如加「文件不存在」）来顺手多认几种失败。
  它是终端 trace 与 ACP 共用的：终端那边拿它决定 verbose 预览行数，改了会连带动终端输出。
  真要扩先单独一轮，把两个入口的期望一起想清楚。
- **不要**在 ACP 那层另写一套「这次算不算失败」的判定。两套口径迟早对不上，
  用户看到终端说失败、编辑器说成功，比不报状态更糟。
- **不要**给 Web Console 的步骤也按 `tool_failed` 变色顺手一起做。字段已经落到
  `live_web_trace.json` 里了，前端怎么画（只变色？还是也折叠？）是 Console 的取舍，
   100|  和本轮的 ACP 载荷没有耦合。
- **不要**给 `locations` 加 `line`。ACP 的 `line` 是 0 基还是 1 基各家实现并不一致，
  拍错了会跳到相邻行；`read_file` 的 `start_line` 是 1 基，直接塞进去大概率差一行。
  要做先核实目标编辑器（Zed）的取值。
- **不要**把搜索类工具的 `path` 也报成 locations。那是扫描范围不是受影响文件，
  一轮 grep 就能在编辑器里挂一串指向工作区根的跳转。
- **不要**在收尾那条重发 `locations`。编辑器按 `toolCallId` 认行，pending 那条报过就在了，
  重发只是多一份载荷；真要改（比如 `write_file` 实际落到了别的路径）得先有「路径变了」这个信号。
- MCP 工具一个 location 都不报：它们的参数里哪个字段是文件路径没法通用地猜，
  与 `pending` 标题那个口子（`_short_tool_target` 认不出 MCP 参数）是同一件事，要做一起做。
   110|- 下一件按 `AGENTS.md` 排队走：**工具 `content` 富化**——现在收尾那条只发纯文本输出，
  编辑器里看不到「这一刀改了什么」；`write_file` / `search_replace` 该发 ACP 的 `diff` 块
  （授权弹窗那边已经在发，`permission.py` 可参照）。难点是 diff 要的旧正文在工具跑完时已经没了，
  得先决定从哪拿（工具返回里带的预览？还是让写工具把 old/new 一起报给入口）。
