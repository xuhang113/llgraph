# 2026-09-09 同一文件改不动：从「无限重试」到「诊断 → 锁定 → 停手」

选题：**稳定性（工具失败恢复 / 死循环）**。接上一轮
`2026-09-08-session-store-durability.md` 结尾点名的两条之一：
「`tool_loop_guard` 目前只拦重复调用，连续失败（同一路径改不动、search_replace
老是 old_string 不唯一）还没有升级策略」。

另一条（跨轮重复读文件，性能线）本轮**没有做**，理由见文末——它有前置条件，
先做会拿稳定性换性能。

## 为什么是这个切口

`tool_loop_guard` 拦的是**参数逐字节相同**的重试。但真实的改码死循环从来不长这样：

```
search_replace(svc.py, old="total = sum(item.price * item.qty ...")   → 未找到
search_replace(svc.py, old="total = sum(i.price * i.quantity ...")    → 未找到
search_replace(svc.py, old="totals = sum(i.price * i.qty ...")        → 未找到
...
```

每次 `old_string` 都差一点（缩进、变量名、上下文各差一点），指纹次次不同，
护栏次次放行。模型手里的原文本来就是错的（凭印象重打、或用了被改前的旧快照），
换着法子猜不会收敛。

先量再改。用真 `apply_edit_hunks` + 真 `compute_blocked_tool_messages` 跑 8 次
近似变体：**8 次全部执行，0 次被拦，没有任何升级**。真正的代价不是那 1803 字符
错误正文，而是每一次重试都是一次**完整 LLM 往返**——`max_turns` 默认 100，
一个犟一点的模型可以把整轮预算烧在一个文件上，最后以
「Sorry, need more steps to process this request.」收场：用户看到的是一次
既没改成、也没给解释的失败。

现有的两层都接不住：`WriteFailureTracker` 只按「连续几次写失败」计数，
文案是通用的分块写建议，不区分路径也不给诊断；`format_apply_failure` 的
单次诊断（试过哪些匹配策略、相近行）质量不错，但**没有跨调用记忆**——
第 6 次失败看到的提示和第 1 次一模一样。

## 做了什么

**新增 `llgraph/core/tool_failure_escalation.py`：按路径而不是按参数聚合失败**

自最近一条真实 user 起，为每个写入路径维护连续失败流水（写成功则清零）。
按连失次数走三级升级，阈值 `agent.json` 的 `agent.edit_failure_escalation_after`
可调（默认 2，`0` / `false` 关闭；锁定 = 阈值 + 2，停手 = 阈值 + 4）：

1. **连失 2 次 → 诊断回灌**。在失败结果末尾追加：已经失败几次、前几次分别试过
   什么（工具 + 失败类别 + `old_string` 首行，最多列 5 条）、以及**按失败类别**
   给的处方。类别从返回正文判定：`old_string` 没匹配上 / 不唯一 / 路径不存在 /
   参数不合法。处方是具体动作而不是「请仔细检查」——例如「没匹配上」给的是
   「先 `read_file(path, start, end)`，从返回正文里逐字符复制一段作 old_string，
   不要重新打字」。
2. **连失 4 次 → 锁住该路径的写工具**。走 `tool_loop_guard` 已有的 ToolNode
   短路径，不执行真实工具。解锁条件只有一个：成功 `read_file` 这个文件。
   这既是断循环，也是把「改之前先看当前原文」变成硬约束。
3. **连失 6 次 → 彻底停写**。读过之后仍然改不动，说明不是「没看原文」的问题。
   此时无视 read 一律拦下，要求模型换落地方式或直接向用户说明卡点。

解锁**不清零计数**，只清「读过了」这一位。所以 read → fail → read → fail
不能无限转：每次失败都往 6 逼近一步。两种模型行为都在有限轮内终止——
不肯重读的停在第 4 次，老实重读的停在第 6 次。

**接线**：`react_tools.build_tool_node` 在 `install_tool_loop_guard` 之后
`install_edit_failure_blocks`（`setdefault` 并入同一张表，loop guard 已有的拦截
优先），在 ToolNode 返回之后 `maybe_annotate_edit_failures`。同步 / 异步两条路
都接。复用 loop guard 的包装意味着拦截仍走写串行门闩的 `mark_done`，
不会让同 path 的后续写等一个已被拦截的前驱。

**一个自己踩的坑，值得记下来**：升级提示是**追加**在真失败正文末尾的，
而拦截占位文案也带 `[llgraph]` 标记。第一版用「正文里含我们的标记就不算失败」
来防止拦截自我放大，结果贴过提示的失败**从此不再计入**，计数永远卡在阈值上，
锁定和停手一次都没触发过——纯函数单测还全绿（用例喂的是没贴提示的裸失败）。
是真 ToolNode 的端到端跑法把它照出来的。改成「**以拦截标记开头**才算占位」，
并把回归用例改成走生产链路（先算拦截、再给失败结果贴提示）。

## 改了哪些路径

- `llgraph/core/tool_failure_escalation.py`（新）
- `llgraph/core/react_limits.py`（阈值解析 `edit_failure_escalation_after`）
- `llgraph/core/react_tools.py`（同步 / 异步两条路接线）
- `tests/test_tool_failure_escalation.py`（新，20 例）
- `docs/项目结构.md`（模块表补一行）

## 怎么验收

回归测试断的是**终止性与解锁语义**，不是文案：

- 失败流水只按路径聚合：另一个文件的失败不串味；写成功清零；新 user 清零
- 拦截占位文案不计入失败（否则拦截自我放大）；但**贴过升级提示的真失败仍要计入**
- 锁定后：读别的文件不解锁、读失败不解锁、读对了才解锁
- 解锁只用一次——读完再失败，计数继续涨，下一次仍被拦
- 停手级别无视 read 一律拦；同一批里别的路径照常可写
- 两种循环都终止：不重读的执行 4 次后停在锁定，每次都重读的执行 6 次后停在停手
- `hint_after=0` 关闭整个机制；`install` 不覆盖 loop guard 已有的拦截

**变异验证**：把「以标记开头才算占位」退回「含标记即占位」，
3 个用例立刻挂，其中循环用例从「执行 4 次后停住」变成 **30 次全部放行**——
正是改前的病。

手工：

- 真 `ToolNode` + 真 `create_filesystem_tools` 端到端跑 8 次近似变体：
  改前 8/8 全部执行、无任何提示；改后第 4~6 次带升级提示、第 7 次起被锁，
  真正落盘操作 6 次（其中 1 次是 `fuzzy(88%)` 正常命中，非误拦）
- `python -m pytest -q` → **578 passed, 3 skipped**（本轮前 558 + 新增 20）
- `python -m ruff check llgraph tests` → All checks passed
- `pip install -e ".[web,terminal]"` 后 `llgraph --help` 33~35ms
  （与上一轮 35ms 同档，新模块只在 ToolNode 链路上，没进启动路径，
  `test_startup_import_budget` 仍绿）；`python -m llgraph --help`、
  `llgraph --list-sessions` 正常

## 未做 / 下一步不要做

- **不要**把升级机制推广到 `run_shell`。shell 的「失败」大量是合法结果
  （跑测试就是要看它红），按失败次数锁工具会把 Agent 变成瞎子。
  要做得先有「命令失败 ≠ 工具失败」的判据，不是本轮这套。
- **不要**把 read 路径也纳进来做「猜路径循环」。那类循环每次换的是 path，
  key 跟着变，本模块按路径聚合的做法天然拦不住；硬做要换成「本问 read 失败总数」
  这种全局键，误伤面大得多。
- **不要**把阈值调小到 1。第 1 次失败之后模型自己改对的比例相当高，
  `format_apply_failure` 的单次诊断已经够用；在第 1 次就贴一大段升级提示
  是纯浪费 token。也**不要**为了省 token 把「已试过什么」那几行删掉——
  让模型看见自己在打转，正是它跳出循环的依据。
- **不要**顺手把 `WriteFailureTracker` 合并进来。它走的是 workspace-context
  注入（下一轮系统提示），本模块走的是工具结果回灌（当场），两条通道的时机
  和缓存语义不同，合并会动到提示词缓存前缀的稳定性。
- 仍然**没有**做「跨轮重复读文件」那条性能线（上一轮已点名）。它依然是真切口，
  但前置条件没变：先要有「被指针引用的历史工具结果不许被 `tool_prune` / 压缩剪掉」
  这个约束，否则省 token 换来丢上下文。下一轮要做它，**先做约束、再做去重**。
- 下一轮建议：上面那条**性能**线（按既定顺序），或**商用体验**里还没碰过的
  「改码命中率」——本轮把改不动的循环拦住了，但没有回答「为什么模型手里的原文
  总是错的」。`search_replace` 前是否该强制带上「你打算改的那段的真实行号」，
  值得单独量一轮。
