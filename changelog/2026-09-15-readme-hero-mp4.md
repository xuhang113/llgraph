# 2026-09-15 README 首页主图改成 mp4

选题：商用体验（循环 gif 换成可播的介绍视频）。

## 做了什么

- 首页上方去掉循环 gif，改成 `react-intro.mp4`（`<video>` + 直链）
- 界面速览仍用下面的 mp4 链接

## 改了哪些路径

- `README.md`
- 删除 `docs/assets/react-intro.gif`

## 怎么验收

- GitHub README 顶部是视频控件，能播多轮 ReAct；没有那张循环 gif

## 未做 / 下一步不要做

- 不要再把 gif 当首页主视觉
