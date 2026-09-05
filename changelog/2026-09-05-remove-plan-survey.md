# 2026-09-05 移除 Plan 编排与问卷交互

选题：商用体验（去掉繁琐 UI）。上一轮补了 changelog 留痕，本轮只拆 Plan / Survey。

## 做了什么

- 删除多 Agent Plan 模式：`llgraph plan`、`/plan`、Web Plan 面板 / 工作流图 / Worker 页、`query_plans`、计划确认弹窗
- 删除可视化问卷：`/survey`、`--no-survey`、`<<<llgraph-survey>>>` 解析、终端向导、Web Survey 对话框
- 提示词不再注入 survey，也不再把 todo 清单说成「那是 llgraph plan」
- Agent 主路径保留：对话、todo_write、spawn_subagent（explore/general）、【规划】行、trace 里「模型决策」步骤、grep/read 内部 plan_grep

## 改了哪些路径

- 删除 llgraph/plan/ llgraph/survey/ prompts/plan/ plan_cli.py plan_service.py survey 相关终端/配置
- 删除 web-ui PlanMainPanel / SurveyDialogs / WorkflowGraph 等
- 改 main.py tools.py agent.py prompt_loader.py meta_commands.py web app.py ConsolePage 等入口
- docs/README 去掉已删入口说明

## 怎么验收

- 终端：`llgraph --help` 没有 plan / --no-survey；会话内 `/plan` `/survey` 不再是内置命令
- Web Console 只有 Agent 会话，侧栏不再出现 Plan 分组或问卷弹窗
- 新建会话 kind=plan 返回 400
- Agent 仍可 spawn_subagent（explore/general）、todo_write、正常改代码
- 会话树 API 只返回 `agents`；前端不再持有 `plans` / `worker` 节点
- `python -m pytest tests/test_prompt_loader.py tests/test_react_limits.py tests/test_agent_turn.py tests/test_session_title_auto.py tests/test_subagent_shared.py`
- `npx tsc -b`（web-ui）

## 未做 / 下一步不要做

- 没有把历史 plan-* 会话目录自动清掉；可当普通会话删
- 不要再加回 Plan 图或问卷向导
- 下一轮接工具层已有方向（入参纠偏 / 路径解析等）即可
