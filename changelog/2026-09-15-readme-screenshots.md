# 2026-09-15 README 补上 Web Console 截图与短视频

选题：商用体验（别人打开 GitHub 首页能看见产品长什么样）。

## 做了什么

- 用本仓库 `llgraph` 工作区拍 Web Console（会话 / 内置工具 / Skills），做成 gif + mp4
- README 首页加上预览；没有拍其它工作区
- GitHub 默认分支改为 `cursor/auto_upgrade`（与定时自动更新同一条线）

## 改了哪些路径

- `README.md`
- `docs/assets/console.png`、`tools.png`、`skills.png`、`console-tour.gif`、`console-tour.mp4`

## 怎么验收

- 打开 README 能看到循环预览图，点开 mp4 能播
- 图里工作区路径是 `/Users/xuhang/Workspace/llgraph`，没有公司仓

## 未做 / 下一步不要做

- 没有录真实跑一轮 Agent 的长视频（会打网关、也更容易带进会话原文）
- 不要把其它工作区的会话截进 README
