# 用真实案例学习：ERCOT 负荷预测项目学习指南

本项目代码和结果全部由助手完成。下面的题目用来把它变成**你自己能解释、能修改**的能力。
流程：先自己作答（中文或英文都可以）→ 再看对应文件核对 → 拿给助手反馈 → 做新题复测。
没有独立完成的部分，不能写成"我掌握了"。诊断题 Q1–Q2 仍在
`preparation/vistra/assessment.md`，建议先做。

## 第 1 课：任务定义与信息可用性（PROTOCOL.md, src/elf/features.py）
1. 用一句英文说明 target、issue time、horizon。为什么 HE01 的 horizon 是 7 h，HE24 是 30 h？
2. A0 假设：18:00 时只知道 D−1 及之前的负荷。为什么 `lag_d1` 不在特征里？如果改用 A1（已知当天到 17:00），哪些特征可以加、哪些仍然不能？
3. 在 `tests/test_core.py` 中找到"注入未来值"测试，用自己的话解释它在证明什么。再设计一个它**测不出来**的泄漏。
4. 为什么 `LAG(load, 24)` 在 DST 日会错？看 `sql/02_features.sql` 中 slot matrix 怎么处理 23/25 小时日。

## 第 2 课：天气预报的时间（reports/weather_semantics.md, src/elf/weather.py）
5. `previous_day1` 是什么？为什么次日 20:00 的目标不能直接用它？手算一个 CDT 例子。
6. JMA 为什么被拒绝？"day2 匹配到只提前 20 h 的 run"为什么意味着泄漏风险？
7. 训练用 ERA5、预测用 GFS（W3）为什么 bias +1121 MW？online bias 校正为什么只能部分修复（W6）？

## 第 3 课：回归与模型（src/elf/models.py）
8. hinge 温度基函数如何让线性回归表达 U 形关系？画出 `max(t−25, 0)` 的形状。
9. 为什么目标用 `y / level7`（relative）？它解决了什么问题，又引入了什么风险？
10. Kalman ridge 中 q 的含义是什么？为什么 q=3e-5 比 3e-4 好？这说明系数漂移有多快？
11. LightGBM 用 L1 objective，评估用 MAE。这两者为什么要一致？

## 第 4 课：评估与结论（reports/benchmark_validation.md, src/elf/compare.py）
12. 为什么用 7 天 block bootstrap，而不是逐小时独立 bootstrap？
13. Zonal 模型 MAE 低 16.5 MW，但最后没有采用。用 CI 解释原因。
14. Chronos-2 单独 1845，加入集成后集成降低 73 MW。用"误差相关性 0.58"解释。
15. 为什么 holdout MAE（1112）比 validation（1310）低？这能说明模型变好了吗？（提示：2024 年 1 月的数据缺口、飓风）
16. 一共跑了 41 次 validation。这对最终数字的可信度有什么影响？locked holdout 起什么作用？

## 第 5 课：SQL 与 BI（sql/, powerbi/MEASURES.md）
17. 独立重写：用 cutoff 前最新版本选取天气预报（issue/valid/available 三个时间）的 SQL。
18. 为什么 fact 表里每个 model 都重复 `y`，在 Power BI 中对 `y` 跨模型求和是错的？
19. **实操（你自己完成）**：在 Power BI Desktop 中导入 `powerbi/data/*.csv`，建 star schema 和 MAE measure，核对 `reference_metrics.csv`。完成后才能写"做过 Power BI report"。

## 第 6 课：聚类（reports/clustering.md, clustering_interpretation.md）
20. raw 聚类的高负荷类占比从 36% 变到 78%，为什么？这对"按年使用同一聚类模型"意味着什么？
21. 为什么 shape k=4 只能叫 exploratory？如果想用 regime 帮助预测，必须怎么做才不泄漏？

## 3 分钟英文讲解（由你自己写稿，助手只给结构）
- Problem & why it matters (1 句)  →  Data & issue-time rules (2 句)  →  Baselines (1 句)
- What worked, with numbers and CI (3 句)  →  One failure and what you learned (2 句)
- Limits & next step (2 句)。不能说 "beat ERCOT" 或 "ranked first"。
