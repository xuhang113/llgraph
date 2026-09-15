# 2026-09-15 README 补回多轮 ReAct 介绍视频

选题：商用体验（首页要有执行过程，且必须是真实 Console/Trace）。

## 做了什么

- 首页主循环改成多轮 ReAct 介绍：Console + Trace 实拍，底下字幕说明第几轮
- 不再用自制海报页（「住在代码仓库里…」那种不是产品 UI）

## 改了哪些路径

- `README.md`
- `docs/assets/react-intro.gif`、`react-intro.mp4`

## 怎么验收

- GitHub README 第一张循环图能看到 Trace 步骤（search / grep / read / 回复）
- 点「多轮 ReAct 介绍」能播 mp4

## 未做 / 下一步不要做

- 没有现场再跑一轮新 Agent（网关当时不通）；画面来自本仓库已有真实 Trace
- 不要再把非产品页面画成「功能截图」
