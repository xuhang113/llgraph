# 2026-09-10 跨轮重复读同一文件：从「原样再读一遍」到「磁盘核对后给短指针」

选题：**性能（token 浪费 / 重复读文件）**。这条线被 `2026-09-08` 与 `2026-09-09`
两轮结尾连续点名，并且都写清了前置条件：

> 「先要有『被指针引用的历史工具结果不许被 `tool_prune` / 压缩剪掉』这个约束，
> 否则省 token 换来丢上下文。下一轮要做它，**先做约束、再做去重**。」

本轮按这个顺序做完了两步。稳定性线的 `tool_loop_guard` / `tool_failure_escalation`
没有推倒重来，跨轮去重是挂在它们旁边的一层。

## 为什么是这个切口

`tool_loop_guard.build_history_index` 的去重窗口从**最近一条真实 user 消息**开始。
用户追问一句「再顺手把 B 也改了」，窗口就清零，模型往往把上一轮刚读过的
同一批文件原样再读一遍。

先量再改。同机构造「3 个真实源文件 × 3 问，每问都重读同一批」的会话，
走真 `ToolNode` + 真 `read_file`，并用 `record_dispatch_prefix`（上一轮做 prompt
cache 时留下的度量）看出站前缀：

| 3 问累计 | 改前 | 改后 |
|---|---|---|
| 每问真实执行的重复 read | 3/3 | 0/3（全部短路） |
| 每问新增工具正文 | 38283 字符 | 867 字符 |
| 第 3 问 state 估算 | 38715 tok | 13771 tok |
| 出站可缓存前缀 | 4% → 20% | 97% |
| **累计未缓存（需全价重算）** | **47937 字符** | **2604 字符** |

代价不止「多了一份全文」。重读会把出站全文预算顶过高水位，触发一次压缩纪元；
Anthropic prompt cache 按精确前缀命中，压缩点之后的全部内容都要重新计费与重算
TTFT——所以真实账单上，一次 8000 token 的重读会连带把前面几万 token 也变成全价。

## 做了什么

### 1. 先做约束：被短指针引用的结果钉住（`context/tool_result_pin.py`）

llgraph 的短路径拦截返回的是「你已在 `tool_call_id=X` 拿到过这个结果」。
这句话只有在 X 的全文**还在上下文里**时才成立。原来没人保证这件事：
X 会被 `prune_stale_tool_messages` 掩码（这一路还会写回 checkpoint 与落盘），
也会被出站压缩成指针，模型于是同时失去正文与重新取正文的入口。

新模块从指针正文里解析被引用的 id，映射到**仍是全文**的 ToolMessage 下标，
`incremental_context` 的两条裁剪路径都把它当钉子。两条硬约束：

- **已压过的不许复活**：`dispatch_compaction` 新增 `compacted_tool_call_ids()`
  供查询，出站钉住时排除。前缀单调性是上一轮 prompt cache 收益的地基，
  一次晚到的引用把历史条目从指针恢复成全文，等于自己击穿缓存。
- **有条数上限**：`context.max_pinned_referenced_tool_messages`（默认 4，0=关闭）。
  指针只增不减，无上限钉住等于把出站预算作废。

### 2. 再做去重：跨轮重复读拦截（`core/cross_turn_read_guard.py`）

只在能证明「重读拿不到任何新信息」时才拦，四条同时成立：

1. **这次 read 真正会返回的行段**已被更早的 read 结果整段覆盖。
   注意不是「请求的行段」：`read_focus` 对大文件不带行段的读返回的是
   「文件头 + 本问检索命中窗」，按全文要求覆盖会让本层几乎永不生效——
   实测正是这个差别把命中率从 1/3 抬到 3/3。多条历史结果可以拼起来覆盖
   （上一轮分两段读完同一个类也算），但它们必须报同一个总行数。
2. **磁盘逐行一致**（`context/read_content_verify.py`）。read 的输出本身就带够
   核对的信息：`--- path (行 s-e / 共 N 行) ---` 头 + `行号| 原文`，
   事后拿正文与磁盘比一遍即可，不必在读的时候额外记指纹。这一条同时覆盖了
   本进程写入、别的进程写入、以及**用户在编辑器里手改**。行数不同、任一行不同、
   文件不存在、路径越界，一律算「变了」。
3. **那些历史结果没被压缩**（查第 1 步的水位）。模型看不见的正文不能拿来当指针。
4. **该路径本会话没有成功写入过**。写过的 read 出站本来就会被
   `stale_read_after_write` 作废，拦下去等于递一份过期原文。

**兜底（这条比上面四条都重要）**：同一文件在同一问里**只拦一次**。
万一模型确实看不见旧正文，它原样再读一次就会真执行——不会出现
「看不见正文又永远读不到」的死转。

**一个自己踩的坑**：兜底第一版是死的。拦截占位会被 `build_history_index`
当成一次成功的 read 记进索引，于是模型的第二次尝试被**本问精确去重**拦住，
放行分支永远走不到。改成「以 `[llgraph]` / `【llgraph` 开头的占位不进历史索引」
（`is_llgraph_placeholder`）。这不影响本问内的重复拦截——那时真结果本身就在索引里。

**上界**：`collect_carry_reads` 从近到远最多回溯 40 条 read 结果。解析是按字符
线性的，长会话不该在每个 tools 节点上无界重扫。实测历史 30 条 read / 97 万字符时
护栏自身 17ms/次（关掉时 0ms），换掉的是一次 LLM 往返 + 几千 token 全价 input。

**开关**：`agent.cross_turn_read_dedupe`（默认开），受 `agent.identical_tool_guard`
总开关约束。

## 改了哪些路径

- `llgraph/context/tool_result_pin.py`（新）、`llgraph/context/read_content_verify.py`（新）
- `llgraph/core/cross_turn_read_guard.py`（新）
- `llgraph/context/incremental_context.py`（两条裁剪路径接钉子）
- `llgraph/context/dispatch_compaction.py`（`compacted_tool_call_ids`、指针标记）
- `llgraph/context/context_settings.py`（`max_pinned_referenced_tool_messages`）
- `llgraph/core/tool_loop_guard.py`（跨轮层接线、占位不进索引）
- `llgraph/core/react_limits.py`、`llgraph/core/react_tools.py`（阈值与 thread_id 接线）
- `llgraph/core/tool_failure_escalation.py`（新占位标记并入拦截判定）
- `tests/test_cross_turn_read_dedupe.py`（新，28 例）、`docs/项目结构.md`

## 怎么验收

回归测试断的是**「什么时候不许拦」**，不是文案：

- 磁盘变了 / 尾部被追加（覆盖行没变但总行数变了）/ 文件被删 → 放行
- 请求行段没被覆盖、两段历史之间有空洞 → 放行
- 历史正文已被出站压缩或已被 checkpoint 掩码 → 放行
- 该路径成功写入过 → 放行；越界与绝对路径不参与核对
- 折叠读：本问新 grep 命中了旧折叠结果没覆盖的区段 → 放行（重读确有新信息）
- 同一文件本问第二次请求 → 放行（兜底）；拦截占位不进历史索引
- `read_files` 必须每个路径都被覆盖才拦
- 钉住：被引用的 read 不再被裁剪 / 压缩；但**已压过的不许复活**；上限与 0=关闭生效

**变异验证**（8 处逐个退回，对应用例必挂）：去掉磁盘核对、去掉总行数一致要求、
允许已压缩条目参与去重、去掉「本问只拦一次」、占位重新计入索引、
折叠读按全文要求覆盖、钉住不排除已压缩、写过的路径也去重。基线 28 例全绿。

手工：

- 真 `ToolNode` + 真 `create_filesystem_tools` 两问端到端（已固化成用例）：
  第一问真读、第二问返回短指针，指针长度不到全文的 1/5
- `python -m pytest -q` → **606 passed, 3 skipped**（本轮前 578 + 新增 28）
- `python -m ruff check llgraph tests` → All checks passed
- `pip install -e ".[terminal,web]"` 后 `llgraph --help` 32~34ms（与上一轮 33~35ms
  同档，新模块只在 ToolNode / 出站链路上）；`llgraph --list-sessions` 正常；
  `test_startup_import_budget` 仍绿

## 未做 / 下一步不要做

- **不要**把跨轮去重推广到 `grep_files` / `glob_files` / 语义检索。read 能拦是因为
  输出里带着「哪个文件哪几行」，可以拿磁盘逐行核对；检索结果的正确性依赖**整个
  工作区**的状态，没有等价的廉价判据，硬做就是拿改码命中率换 token。
- **不要**去掉磁盘核对改成「按 mtime/size 判断」。同一秒内的写、`mtime` 被工具
  重置、跨机挂载时间偏移都会骗过它；逐行比一遍在 2MB 以内只要几毫秒，
  这点成本换的是「模型手里的原文一定是真的」。
- **不要**把「本问只拦一次」的兜底改成 2 次以上或干掉。它是本层唯一的活口：
  一旦覆盖判定或水位判定有 bug，兜底能把死转变成「多读一次」。
- **不要**把 `max_pinned_referenced_tool_messages` 往大调（比如 16）。钉住的是
  **不占压缩预算**的全文，调大等于让出站在长会话里失去收缩能力；
  真需要更多历史正文，该走的是 `search_session_history` 而不是钉住。
- **不要**顺手把 `filesystem_tools` 的读结果格式改「更好解析」（例如加 JSON 头）。
  本层刻意只依赖现有格式，改格式会同时动到 prompt cache 前缀、
  `stale_read_after_write`、`read_segment_dedupe` 三处的解析。
- 一个已知但**本轮刻意不修**的小问题：`dispatch_compaction.tool_content_is_compact`
  是按正文里**含**某些标记判定的，读 llgraph 自己的源码（里面就写着这些标记字符串）
  会被误判成「已经是短指针」。用真实业务仓库不会撞上，改判定要动出站压缩的
  纪元语义，值得单独一轮，并且要先补「误判会怎样」的量化。
- 下一轮建议：**性能**里还没碰过的「无效工具调用 / 上下文膨胀」——例如
  同一问里 grep 同一 pattern 只是换 `path` 的扩散式检索；或者**商用体验**里
  上一轮点名的「改码命中率」：`search_replace` 前是否该强制带上目标段的真实行号。
