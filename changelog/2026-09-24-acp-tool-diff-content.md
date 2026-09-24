# 2026-09-24 这一刀改了什么：ACP 工具 `content` 带 diff 块

选题：**排队第 2 件（编辑器里干活）**。`2026-09-23` 失败态那篇末尾点名的就是这件：

> 「**工具 `content` 富化**——现在收尾那条只发纯文本输出，编辑器里看不到「这一刀改了什么」；
> `write_file` / `search_replace` 该发 ACP 的 `diff` 块（授权弹窗那边已经在发，`permission.py` 可参照）。
> 难点是 diff 要的旧正文在工具跑完时已经没了，得先决定从哪拿。」

在此之前，编辑器里一次 `search_replace` 跑完只留下一行 `已替换 app.py（1 处）`，
一次 `write_file` 只留下 `已写入 app.py（1234 字符）` 加前 40 行快照——
    10|「改了什么」得自己去翻文件、或者回想改之前长什么样。而**同一次改动**在授权弹窗里
是有 diff 的（用户点「允许」时看得清清楚楚），批准完反而看不到了。

## 做了什么

### 1. 旧正文从哪拿：写工具落地那一刻报一次

diff 要的改前 / 改后全文，只有写工具自己手里有：`search_replace` 读完文件才应用 hunk，
`write_file` 覆盖前才读得到旧的那份。跑完这些都没了，trace 步骤里剩的只是工具返回的那段话。
另外两条路都不行：

    20|- **从工具返回里解析预览**：`format_apply_success` 的输出是给模型看的摘要，
  不是完整的改前 / 改后；靠正则从里面反推 diff，工具文案一改就错位。
- **从授权弹窗那条拿**：弹窗在写**之前**发，可能被拒；而且只有 ACP 注册了闸门时才有，
  `--write`（免确认）那条路一次弹窗都不弹，正好是最常用的那条。

所以在写真正落地之后（`_persist_text` 返回之后）报一次，与 `2026-09-22` 的起步通知同一个形状：
ContextVar 上的观察者，没人登记就是空操作，CLI / Web Console 一行行为都没变。
落空的写自然一条都不报——校验没过、`old_string` 没匹配上、用户点了拒绝，都走不到那一行。

### 2. 归属：ToolNode 的调用包装顺手记下当前 tool_call_id

    30|写工具只收到参数，拿不到 `tool_call_id`，可入口正是按这条 id 把 diff 挂回编辑器里那一行的
（并行写两份文件时，少了它就分不清哪份属于哪次调用）。链路上唯一「一次调用从头包到尾」
的位置还是 `tool_invoke_timing` 的那层包装——起步通知已经搭在那儿，这轮把 id 记进同一个括号里的
ContextVar，工具内部读得到。**没有再叠一层包装**：两层包装顺序一乱，归属就可能记到隔壁调用上。

### 3. 发在哪：收尾那条的 `content` 里，排在文本前面

diff 攒到这次调用收尾时跟状态一起发，不单发一条 `tool_call_update`：
调用还没结束，多发一条只是让编辑器里那一行多刷新一次。

`content` 的顺序是 **diff 在前、纯文本输出在后**。文本没有删掉：
    40|语法诊断、分块提示、`取自未保存的缓冲区` 那句注记都在里面，砍了就少一块信息；
但第一眼该看到的是改动本身。新建文件的 `oldText` 发 null（ACP 据此画成整份新增），
与 `permission.py` 同一个写法——同一次改动在弹窗里和结果里长得不一样是最容易让人怀疑的。

四个「不发」：

- **正文没变的写入不发**：编辑器里一个空 diff 只是噪声。
- **认不出归属的不发**：没有 `tool_call_id` 就挂不上任何一行。
- **补不出绝对路径的不发**：与 `locations` 同一个口径（ACP 只认绝对路径）。
- **改前 + 改后超过 `MAX_DIFF_CHARS`（20 万字符）的不发**：生成的文件动辄几十万字符，
  一条更新能把 stdio 通道堵住，而那种文件的逐行 diff 本来也没人读。此时那次调用仍有文本输出。
    50|
## 改了哪些路径

- `llgraph/core/tool_progress.py`（`ToolEdit`、编辑观察者、`use_current_tool_call`、`notify_file_edited`）
- `llgraph/core/tool_invoke_timing.py`（调用包装里登记当前 `tool_call_id`）
- `llgraph/core/filesystem_tools.py`（`write_file` / `append_file` / `search_replace` 落地后报改动）
- `llgraph/editor/acp/updates.py`（`tool_call_diffs`、`absolute_workspace_path`、`tool_call_from_step(diffs=)`）
- `llgraph/editor/acp/sink.py`（`tool_edited` 按调用攒改动，收尾时一起发）、`turn.py`（登记观察者）
- `tests/test_acp_tool_diff.py`（新，25 例）、`tests/test_acp_end_to_end.py`（+1 例）
- `README.md`、`docs/模块说明.md`、`docs/项目结构.md`、`AGENTS.md`、`.cursor/rules/cloud-agent.mdc`
    60|
## 怎么验收

- `python3 -m pytest tests -q`（除 `test_prompt_cache_breakpoints.py`）→ **1078 passed, 10 skipped**
  （本轮前 1052；新增 26 例）；`ruff check llgraph tests` 通过。
  `tests/test_prompt_cache_breakpoints.py` 单跑仍是 **11 failed**，与本轮无关且与上一轮同因：
  本机装的 `langchain-anthropic` 里 `_format_messages()` 多了必填关键字参数 `model`。
  skip 全是本机没装的可选依赖（`fastapi` / `lancedb` / `langchain-openai` / `langchain-ollama` /
  `langchain-google-genai`）
- 通知语义：没登记观察者时空操作、作用域随 with 块结束、观察者抛异常不外溢；
    70|  没有当前调用 id / 空路径 / 正文没变一律不报
- 归属：`use_current_tool_call` 在工具内部读得到那条 id（同步与 async 两条路都测），
  出了这次调用即清空
- 写工具：`write_file` 新建报 `old_text=""`、`search_replace` 报整份改前 / 改后、
  `append_file` 报合并后的全文；`old_string` 没匹配上那次与被拒那次一条都不报（文件也没动）
- 载荷：相对路径补成绝对、新建文件 `oldText` 为 null、没有工作区根时不发、
  超长不发、正文没变不发、脏输入（非列表 / None / 空 dict）不炸；
  `content` 里 diff 排在文本前面，只有 diff 时也成 `content`，两样都没有时没有 `content` 这个键
- Sink：改动按 `tool_call_id` 挂回对应那条收尾更新；并行两次调用各归各位；
  同一次调用改同一份文件两回合成一条（最早的改前 + 最后的改后）；
    80|  已发过的改动不会在下一条步骤上重发；不属于任何已知调用的改动被丢掉
- 端到端（stub server 说 Anthropic 协议，**真跑 ReAct**，`allow_write=True`）：
  模型改 `app.py` 那轮，收尾那条 `tool_call_update` 的 `content[0]` 就是 `diff` 块，
  `path` 是绝对路径、`oldText` / `newText` 分别是改前改后全文，`content[1]` 仍是原来的文本输出
- `pip install -e .` 后 `llgraph --help`、`llgraph acp --help`、`python3 -m llgraph --help` 正常

## 未做 / 下一步不要做

- **不要**把 diff 改成单发一条 `tool_call_update`（改完立刻推）。一次调用在编辑器里就该只有一行在动，
  收尾那条本来就要带 content，多发一条只是多一次刷新；真要做得先想清楚「改了三个文件」在编辑器里画成几条。
    90|- **不要**为了省字把 diff 换成 unified diff 文本塞进 `content` 块。ACP 的 `diff` 块是编辑器唯一会
  渲染成改动预览的类型，换成纯文本就退回这轮之前的样子了。
- **不要**因为有了 diff 就把工具的文本输出删掉。语法诊断、分块提示、`取自未保存的缓冲区`
  那句注记都在里面，删了这些信息没有别的地方能看到。
- **不要**把 `MAX_DIFF_CHARS` 调大或改成「截断成前 N 行的假 diff」。截断的 diff 会让人以为
  后面没改动，比不发更糟；真要处理大文件得先有「按 hunk 发」这个能力。
- **不要**给 `delete_file` / shell 里的改动补 diff。删除在 ACP 里没有对应块（`oldText` 全文 + 空 `newText`
  会被画成「清空文件」而不是删除），shell 改了什么则根本不在 llgraph 手里。
- **不要**把授权弹窗那条的 `perm_*` id 合并到工具级 id 上顺手做掉。与上一轮同一个理由：
  一次调用可能问多次授权，合之前得先决定编辑器里画成几行。
   100|- 下一件按 `AGENTS.md` 排队走：**ACP 终端**（`terminal/create` + `terminal/output`）——
  `run_shell_command` 现在要等命令跑完才一次性回填输出，编辑器里看不到滚动中的日志；
  难点是客户端不一定声明 `terminal` 能力（没有时必须回落现有写法），
  且输出要边跑边推，得先决定从工具的哪一层拿到流式 stdout。
