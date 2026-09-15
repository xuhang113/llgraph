# 2026-09-15 README 缩小预览并去掉已删除的 Plan

选题：商用体验（首页别撑满，也不要展示已下线的 Plan）。

## 做了什么

- 首页只留一张小介绍 gif（宽 420）+ 一排缩略图；不再把竖图界面 gif 铺满
- 文案去掉 Plan：控制台 / 删会话 / 目录树不再写 `plan/`、`llgraph plan`
- 重新构建 `web-ui/dist` 后重拍会话 / 工具 / Skills（源码里 Plan 早删了，旧 dist 还带着 New Plan）

## 改了哪些路径

- `README.md`
- `docs/assets/console.png`、`tools.png`、`skills.png`、`console-tour.gif`、`console-tour.mp4`、`react-intro.gif`、`react-intro.mp4`

## 怎么验收

- GitHub README 图不大，侧栏只有 New Agent，没有 New Plan / Plan 0
- README 全文搜不到 Plan 产品入口

## 未做 / 下一步不要做

- 没有改删除接口里对历史 `plan-*` 目录的兼容
- 不要再把 Plan 按钮加回 Web Console
