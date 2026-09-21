# 2026-09-21 读到你眼前那版：ACP `fs/read_text_file` / `fs/write_text_file`

选题：**排队第 2 件（编辑器里干活）**。`2026-09-20` 续聊那篇末尾点名的就是这件：

> 「下一件按 `AGENTS.md` 排队走：**`fs/read_text_file` / `fs/write_text_file` 反向请求**——
> 接了就能读到编辑器里**未保存的缓冲区**，比我们自己读磁盘准；难点是文件工具会多一条来源，
> 得想清楚哪些路径走编辑器、哪些仍走磁盘（沙箱与权限判定都在磁盘那条上）。」

在此之前，用户在编辑器里改了几行还没按保存，llgraph 读到的仍是磁盘上那版旧的：
模型据此改代码，改完还可能被用户随手一次保存整段盖掉。

## 做了什么

### 1. `core/editor_fs.py`：文件工具的第二条来源

与 `permissions/approval.py` 的授权闸门同一个形状：一个 ContextVar，`run_acp_turn`
在 invoke 外面登记，工具在 LangGraph 线程池里跑也拿得到（langchain 提交任务时复制 context）。
**没登记来源时一律走磁盘**，所以 CLI / Web Console 的行为一个字节都没变。

分工划清：**编辑器只回内容，不回「这个路径能不能动」。** 路径解析、沙箱、越界、
只读模式、文件类型、大小上限全部仍在磁盘那条上；缓冲区只在「读出来的正文」这一层接进来。

### 2. 「有未保存改动」才改道，这是本轮唯一的行为开关

`filesystem_tools._editor_text()` 拿到缓冲区后先与磁盘正文比一次，**一致就当没有缓冲区**：

- 读：不挂提示。不然每次 `read_file` 都多一句噪音，模型还会以为文件有问题。
- 写：不绕编辑器。绝大多数文件在编辑器里是干净的，照旧走 `atomic_write`
  （原子替换、保权限位、保符号链接）——那些语义不该为了这个特性丢掉。

只有**脏文件**（缓冲区 ≠ 磁盘）才两头都改道：

| 工具 | 改前正文取自 | 落地方式 |
|------|--------------|----------|
| `read_file` / `read_files` | 缓冲区 | —（返回里注明「取自未保存的缓冲区」） |
| `search_replace` | 缓冲区 | `fs/write_text_file` |
| `append_file` | 缓冲区 | `fs/write_text_file` |
| `write_file` | 缓冲区（仅用于 diff 预览） | `fs/write_text_file` |

**为什么脏文件必须交回编辑器写**：我们自己原子替换磁盘，编辑器里那份未保存的缓冲区还在，
用户过一分钟按一次保存，llgraph 这一刀就被整段盖掉了，而模型以为改成了。
交回编辑器写，改动落在用户眼前那份上（Zed 会顺带保存）。

**为什么 hunk 也要按缓冲区匹配**：`search_replace` 若按磁盘正文匹配、再把结果交出去，
等于用旧内容覆盖用户没保存的编辑。按缓冲区匹配则是「在你现在这版上改」。

编辑器写不下（没这个能力 / 回了错）就退回自己落盘，并在返回里说明已直接落盘——
这次修改不能因为编辑器抽风而丢掉。

### 3. `editor/acp/fs_bridge.py`：两条反向请求

与授权弹窗共用 `jsonrpc` 那条反向链路（工作线程发出并阻塞，读循环收回包唤醒）。三个决定：

- **只在 `initialize` 声明了 `clientCapabilities.fs.readTextFile` / `writeTextFile` 时才建桥**，
  读写能力分别判。没声明的编辑器一条 `fs/*` 都不会收到。
- **只对工作区内的绝对路径发请求。** `~/.llgraph/skills|rules` 这类外部读走磁盘：
  客户端只认自己打开的项目，问过去也只会被拒，白等一个超时。
- **连着失败 3 次就熔断**，本会话之后只走磁盘。超时给 15 秒（不像授权弹窗要等人点按钮，
  这一步没有人参与）；一轮里十几次读文件，每次都赔一个超时会把对话拖死。
  中间成功一次即清零，短暂抖动不该永久降级。

## 改了哪些路径

- `llgraph/core/editor_fs.py`（新）
- `llgraph/editor/acp/fs_bridge.py`（新）
- `llgraph/core/filesystem_tools.py`（读路径 + 三个写工具接上来源，抽出 `_editor_text` / `_persist_text`）
- `llgraph/editor/acp/server.py`（握手记下客户端 fs 能力、按会话建桥）、`turn.py`（登记来源）
- `tests/test_acp_fs_bridge.py`（新，25 例）、`tests/test_acp_end_to_end.py`（+2 例）
- `README.md`、`docs/模块说明.md`、`docs/项目结构.md`、`AGENTS.md`、`.cursor/rules/cloud-agent.mdc`

## 怎么验收

- `python3 -m pytest tests -q` → **1030 passed, 10 skipped**（本轮前 1003；新增 27 例。
  skip 全是本机没装的可选依赖：`fastapi` / `lancedb` / `langchain-openai` / `langchain-ollama` / `langchain-google-genai`）；
  `ruff check llgraph tests` 通过
- 来源语义：没登记来源时 `editor_buffer_text` 回 None、`editor_write_text` 回 False（CLI / Web 不变）；
  作用域随 with 块结束；来源抛异常或回非字符串都当「没有这条来源」
- 文件工具：脏文件 `read_file` / `read_files` 读到缓冲区且带提示；干净文件无提示；
  缓冲区超过 `max_read_bytes` 被拦（磁盘 stat 拦不住它）；
  脏文件 `search_replace` / `append_file` / `write_file` 交给编辑器写且磁盘保持原样；
  编辑器拒收时回落磁盘并说明；干净文件与新建文件照旧原子落盘、一次都不问编辑器
- 桥：请求载荷带 sessionId 与绝对路径；工作区外路径与未声明的能力不发请求；
  回包缺 `content` 判失败；连续 3 次失败后 `disabled` 且不再发请求，中间成功一次清零；
  本轮已取消时不发请求，等待期间被取消不计进熔断
- 协议层（真 pipe，测试侧扮编辑器）：声明 fs 能力后收到 `fs/read_text_file`（sessionId / path 正确），
  回的 content 送达工具；没声明能力时一条 `fs/*` 都不发、`editor_files` 为 None
- 真 ReAct（stub server 说 Anthropic 协议）：读那轮模型收到的工具结果是缓冲区那版而不是磁盘版；
  改脏文件那轮改动落在编辑器上（含用户未保存的那行），磁盘保持原样
- `pip install -e .` 后 `llgraph --help`、`llgraph acp --help`、`python3 -m llgraph --help` 正常

## 未做 / 下一步不要做

- **不要**把 grep / glob / 代码索引也接到编辑器上。那几条靠 ripgrep 扫整个工作区，
  逐文件问编辑器等于把一次进程调用换成上千次 JSON-RPC 往返。真要做得先有批量方法。
- **不要**在缓冲区与磁盘一致时也发 `fs/write_text_file`「保持一致」。那会让编辑器把干净文件
  标脏 / 触发保存钩子（格式化、LSP 重启），还丢掉原子写与权限位。
- **不要**把缓冲区读进 `cross_turn_read_guard` 的「磁盘逐行未变」判定里当真相：
  那个守卫比的是磁盘，混进缓冲区会让它误判「没变过」。本轮没碰它。
- **不要**去掉「一致就当没有缓冲区」这一步改成「有能力就全部改道」。那样每次读都会挂提示，
  每次写都会绕开原子落盘，收益为零、风险全留。
- 编辑器里**只存在于缓冲区的新文件**（还没保存过、磁盘上没有）读不到：
  磁盘那条先判文件不存在就返回了。要支持得先决定「文件存不存在」谁说了算，那是动路径解析。
- 没做局部读（ACP 的 `fs/read_text_file` 支持 `line` / `limit`）：现在每次取全文再自己切，
  大文件比走磁盘多传一份正文。真要优化先量一遍长文件的往返字节。
- 熔断阈值（3 次 / 15 秒）是拍的，没有按往返耗时自适应。
- 下一件按 `AGENTS.md` 排队走：**`tool_call` 的中间态**——现在 `sink.py` 只在工具跑完
  才推一条完成态 `tool_call`，编辑器里一个长命令（跑测试）期间看不到任何动静；
  要补 `pending` → `in_progress` → `completed` 三段更新，难点是 trace 那边现在只在步末给事件，
  得先有「步开始」这个钩子，且同一个 toolCallId 要在三段之间稳住。
