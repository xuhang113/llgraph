# 2026-09-15 Shell 权限闸门改成按简单命令判定

选题：稳定性（兼顾商用体验）。`llgraph/permissions/shell.py` 原本用两条正则扫整条命令，
边界只认「行首或 `;` `&` `|` 之后」，两头都不准：

- 误伤：只读模式下 `grep ... 2>/dev/null`、`pytest 2>&1 | tail`、`cat < README.md`、
  `echo 'a > b'`、`git log --format='%h -> %s'` 全被「禁止 shell 重定向」拦掉。
  只读是默认模式，Agent 每次撞上都白烧一轮工具调用。
- 漏放：`ls\nrm -rf build`、`echo $(rm -rf build)`、反引号、`bash -c 'rm -rf build'`、
  `find . -exec rm {} \;` 都能绕过只读黑名单；`sudo rm -rf /`、`rm -r -f /`、`rm -rf /*`
  连 `-w` 模式的高危拦截也躲得过。

## 做了什么

- 新增 `llgraph/permissions/shell_lex.py`：把命令切成「简单命令 = 词 + 重定向」序列。
  处理引号与转义、`;` `&&` `||` `|` 换行 `(...)` 分隔、fd 前缀（`2>`）、fd 复制（`>&1`）、
  `$(...)` / 反引号下钻、heredoc 正文跳过（正文是数据不是命令）。
- 重写 `check_shell_command`：逐条简单命令判定。
  - 只读：命令名黑名单（`rm`/`mv`/`cp`/`mkdir`/`touch`/`tee`/`dd`/`chmod`… ）、
    `git`/`npm`/`pip`/`uv`/`cargo`/`apt` 等写类子命令、`sed -i` 就地改写、
    `find -delete` / `find -exec rm`、`sudo` 提权、以及**写入类重定向**。
    `2>&1`、`2>/dev/null`、`< file`、`<<<`、引号里的 `>` 一律放行。
  - 任意模式：递归删系统根目录（`/`、`/*`、`~`、`$HOME`、`/etc`…，含 `-r -f` 分写与
    `--no-preserve-root`）、写块设备（`dd of=/dev/sda`、`> /dev/nvme0n1`）、`mkfs*`、fork 炸弹。
  - `sudo` / `env` / `nohup` / `timeout` / `xargs` 剥壳到真实命令；`bash -c '<cmd>'` 递归判定。
- 新增 `tests/test_shell_permission_gate.py`：只读放行 19 条常见只读命令、
  只读拦住 20 条写/绕过写法、11 条任意模式高危、8 条 `-w` 下应放行的作用域内写操作，
  外加解析层单测（fd 前缀、`/dev/null`、引号内 `>`、heredoc）。
- `docs/操作手册.md` §5.2 同步实际策略。

## 改了哪些路径

- llgraph/permissions/shell_lex.py（新增）
- llgraph/permissions/shell.py
- tests/test_shell_permission_gate.py（新增）
- docs/操作手册.md

## 怎么验收

- `python3 -m pytest -q` → 762 passed, 4 skipped（原 697 passed）
- `python3 -m compileall -q llgraph` 通过；`pip install -e .` 与 `llgraph` 入口未动
- 只读模式手验：`pytest -q 2>&1 | tail -20` 放行，`echo hi > out.txt` 拦截，
  `ls\nrm -rf build` 拦截

## 未做 / 下一步不要做

- 没有做真沙箱：`python3 -c "open('x','w')"`、`node -e` 这类解释器内写盘仍拦不住，
  该由 `llgraph/sandbox/` 的 OS 沙箱负责，不要在正则/词法层继续堆规则。
- 没碰 `llgraph/sandbox/`、`shell_tools.py` 的执行链与 `shell_cwd` 的 cd 解析。
- 下一轮建议：`llgraph/permissions/mcp.py` 的写类 MCP 工具判定还是关键词命中
  （`name + description` 里出现 write/edit/delete 就算写），误判面大且零测试；
  或补 `llgraph/gateway/`、`llgraph/commands/` 的零覆盖核心路径。
