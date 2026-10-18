# Related work：电力负荷预测的公司、竞赛与研究做法（助手整理，2026-10-13）

目的：学习别人怎么做，并在本项目的固定协议（PROTOCOL.md）下实现我们自己的版本。
每条注明来源类型和可信度；数字只在原文给出时引用。不同数据/horizon/指标的数字**不能直接比较**。

## 1. 运营方 / 公司团队

| 来源 | 做法要点 | 对我们的启发 |
|---|---|---|
| **ERCOT Mid-Term Load Forecast (MTLF)**（[ERCOT load forecast page](https://www.ercot.com/gridinfo/load/forecast/)，[2022 overview](https://www.ercot.com/files/docs/2022/08/09/6%20Load%20Forecasting%20Overview.pdf)） | 多个 vendor 模型并行，每小时发布 168 h 预测，标记当前选用的模型；输入为天气预报 + calendar；分别报告 forecast error 与用实际天气的 **backcast** error。公开的月度 Day-Ahead RUC MAPE（2024–2025）约 **1.4%–3.5%**（`DayAhead-Backcast-ReportSep25.xlsx`）。 | (1) 多模型 + 选择/组合；(2) forecast vs backcast 分开报告——本项目 WX 就是 backcast 参照。ERCOT 数字是**外部参照**：它使用当天实时负荷、商业天气预报，发布时点（约 14:00）与我们不同，不是同协议比较。 |
| **EDF R&D**（de Vilmarest & Goude，IEEE 竞赛冠军） | 每小时单独建模；GAM 为核心；天气预报统计校正；Kalman filter 在线调整系数；online expert aggregation（R `opera`）。 | 已实现：per-hour Ridge（GAM 风格 hinge 温度项）+ Kalman 在线更新（K 系列）+ 集成（E 系列）。待做：天气预报校正（MOS）。 |
| 公用事业常用实践（Itron MetrixND、ISO 文档等，一般性知识，未逐条核实） | 回归为主（温度非线性 + calendar + 交互），按小时/日型分模型，配合预测员人工调整。 | 解释性强的 Ridge/GAM 仍是面试中最好讲的主线；树模型和 TSFM 作为对照/集成成员。 |

## 2. 公开竞赛

| 竞赛 | 任务与赢家做法 | 启发 |
|---|---|---|
| **IEEE DataPort Post-COVID Day-Ahead**（2021；[论文](https://eprints.gla.ac.uk/254028/1/254028.pdf)；冠军 [arXiv 2110.00334](https://arxiv.org/abs/2110.00334)） | 08:00 issue，16–40 h ahead，提供天气预报，MAE 排名。冠军：天气预报线性校正（温度 MAE 3.00→1.69 °C）、per-hour AR/Linear/GAM/RF/MLP、**intraday correction**、**Kalman 自适应**（线性模型 MAE 22.8→11.2，最大收益）、在线聚合。前 5 名中 3 队用多模型集成；30 天评估期内多名次统计上分不开。 | 自适应 > 模型复杂度；集成稳定有效；评估期短时要做 bootstrap/DM 检验（我们用 block bootstrap）。 |
| **GEFCom2012/2014/2017**（Tao Hong 等；[GEFCom2017 overview](https://blog.drhongtao.com/2019/04/gefcom2017-hierarchical-probabilistic-load-forecasting.html)） | 2014：概率预测（99 quantiles，pinball loss）；"Vanilla" benchmark = 线性回归（trend + calendar + 温度多项式及交互）；**recency effect**：前若干小时温度的 lag 与移动平均（[Wang, Liu & Hong 2016](https://ideas.repec.org/a/eee/intfor/v32y2016i3p585-597.html)）。2017 冠军：数据可视化清洗 + 多模型 + 每个对象选择/组合。 | 我们的 `w_tlag*`、`w_tsmooth` 就是 recency effect；Vanilla 式回归是可讲清楚的 baseline。 |
| **BigDEAL Challenge 2022**（[blog](https://blog.drhongtao.com/2022/12/bigdeal-challenge-2022-final-leaderboard.html)，[一支高排名队伍 arXiv 2305.05575](https://arxiv.org/abs/2305.05575)） | 峰值负荷预测（幅度、时间、形状）。高排名方案：LightGBM + 趋势建模 + 特征选择（温度/日历的 cluster permutation）+ temporal hierarchies 组合。 | 我们的 LightGBM + 相对 level 目标处理趋势；峰值误差是辅助指标；temporal hierarchies 是待试方向。 |
| **Energy-Arena**（[arXiv 2604.24705](https://arxiv.org/abs/2604.24705)） | 活跃前瞻 leaderboard（德国/奥地利），按 gate closure 提交，避免回测调参。 | 我们的 locked holdout 是离线版本的同一理念；之后可做迁移项目（需授权）。 |

## 3. 学术研究方向

### 3.1 经典统计/ML（仍是强 baseline）
- GAM / 分段线性温度响应、per-hour 模型、recency effect、holiday 处理——多次竞赛中仍是第一梯队。
- 梯度提升树（LightGBM）+ 特征工程：竞赛常见赢家组件。
- [ERCOT 天气与时间特征研究 arXiv 2310.08793](https://arxiv.org/abs/2310.08793)：全部特征一起用不一定最好，特征选择/ablation 很重要。

### 3.2 深度学习与 benchmark
- [KIT grid-level load benchmark, arXiv 2607.15705](https://arxiv.org/html/2607.15705)（2026）：在 TransnetBW 等数据上，经充分调参的标准 Transformer 优于 LightGBM、N-HITS、TFT 及 Chronos-2；但他们使用 ERA5 再分析（作者承认实际预测时不可得）且假设负荷立即可得——**协议比我们宽松**。
- 启发：深度模型在大量数据 + 充分调参时可能领先，但需要更多算力与严格协议；不预设它一定更好。

### 3.3 Transfer learning / Foundation models（新方向）
- **Chronos-2 / Chronos-Bolt / Moirai / TimesFM**：零样本时间序列 foundation models。
  - [ERCOT TSFM benchmark, arXiv 2602.10848](https://arxiv.org/html/2602.10848)：只用负荷历史（无天气），24 h horizon 上 Chronos-Bolt/Chronos-2/Moirai-2 的 MASE 约 0.31–0.42，显著好于 Prophet/SARIMA/seasonal naive；没有训练好的 LightGBM 对照，也没用天气。
  - [Exogenous data in TSFMs, arXiv 2602.05390](https://arxiv.org/abs/2602.05390)：Chronos-2 和 TTM 能利用外生变量；在新加坡数据上简单 LSTM 常胜过所有 TSFM——"不是普遍更优"。
- **BuildingsBench**（[NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3f17bf868966df01ca125e5bbc9ee24e-Abstract.html)）：90 万合成建筑预训练，零样本迁移到真实建筑；fine-tune 对多数建筑有帮助。偏建筑级，不是系统级。
- **我们的实现**：T1/T2 在相同 cutoff 下零样本测试 Chronos-2（无/有 GFS 温度 + calendar 协变量），可作为集成成员。若有效，下一步是 fine-tune 或 covariate 设计。

### 3.4 LLM 用于负荷预测
- Time-LLM 等 "reprogramming" 方法、[TSLLM-Load zero-shot, arXiv 2502.16896](https://arxiv.org/abs/2502.16896)、加入社会/新闻文本的 LLM（[News-driven load forecasting](https://www.engineering.org.cn/engi/EN/10.1016/j.eng.2026.02.031)）——多在家庭/小区数据上报告提升，与系统级 day-ahead 强 baseline 的公平比较很少。
- [LLMs in the electric energy sector, arXiv 2403.09125](https://arxiv.org/abs/2403.09125)（Joule 2024）：用 ERCOT 数据评估 LLM 做负荷/电价预测，指出需要领域数据、工具调用与 RAG。
- 现实判断：对系统级逐小时负荷，**数值预测由 LLM 直接生成**通常不如专用模型。LLM 更合理的角色：(1) 把事件文本（飓风、寒潮、停电新闻）转为结构化特征——但历史新闻的发布时间必须早于 cutoff，否则泄漏；(2) 解释预测和误差（报告生成），数字来自结构化计算。可作为后续独立项目，而不是本项目的主模型。

## 4. 已经或计划在本项目实现的版本

| 想法 | 来源 | 状态 |
|---|---|---|
| Issue-time 安全的天气预报输入 | ERCOT/竞赛 | 已实现（GFS vintages，per-hour 规则） |
| Per-hour 回归 + 非线性温度 | GEFCom Vanilla / EDF GAM | 已实现（RidgeWeather） |
| Recency effect 温度动态 | Wang, Liu & Hong 2016 | 已实现（W5） |
| Kalman 在线自适应 | de Vilmarest & Goude | 已实现（K 系列） |
| 在线偏差校正 | intraday/online correction 思路 | 已实现（OnlineBias） |
| 模型集成 | 多个竞赛赢家 | 已实现（E 系列，含 online 权重：E7–E9） |
| Day-Ahead Market 价格特征 | ERCOT 市场结构（本项目原创尝试） | 已测试：收益很小（D1/E3） |
| Foundation model 零样本 | Chronos-2 | 已实现（T1–T4）；单独较弱（1845），作为集成成员 −73 MW |
| Foundation model fine-tune（transfer learning） | Chronos-2 LoRA，BuildingsBench 思路 | S2（GPU）：修复 dropout bug 并取 3 seed 平均后单独 1325（最好的单模型）；集成 E11 1247（vs E9 −62），再加强成员也不再变好（集成饱和） |
| 天气预报 MOS 校正 | EDF 冠军方案 | 待做 |
| Temporal hierarchies / 峰值专门模型 | BigDEAL 2022 | 待评估 |
| 分区（8 weather zones）建模后求和 | ERCOT zonal 结构 | 已测试（Z1/E10）：没有显著收益 |
| LLM 事件特征 | 新研究 | 仅作未来方向，需严格发布时间 |
