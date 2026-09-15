# 2026-09-15 README 补 Trace 截图与 ReAct 介绍片

选题：商用体验（首页能看见过程，而不只是静态窗）。

## 做了什么

- 用本仓库 `llgraph` 工作区拍右侧 **Trace**（多轮：模型决策 → search/grep/list/read → 回复）
- 做成介绍片：先讲 llgraph 是什么、一次对话如何 ReAct，再切入真实 Trace / 会话 / 工具 / Skills
- README 首页改成介绍片作主图，保留原来的界面速览 gif

## 改了哪些路径

- `README.md`
- `docs/assets/trace.png`、`trace-rounds.png`、`trace-react.png`、`console-features.png`
- `docs/assets/react-intro.gif`、`react-intro.mp4`

## 怎么验收

- 打开 README 先看到介绍循环图，点 mp4 能播完整介绍
- Trace 图里能看出多轮工具，工作区是 llgraph，没有公司仓

## 未做 / 下一步不要做

- 当时网关 DNS 不通，没有再现场打一轮新的 Agent；Trace 用的是本仓库已有真实会话
- 不要把其它工作区的会话截进 README
