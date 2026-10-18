# Model card — E9: ERCOT day-ahead hourly load ensemble

Written 2026-10-13. Personal research project on public data; not an
operational system, not affiliated with ERCOT or Vistra. **No public leaderboard exists for this
task**; all claims are relative to the fixed internal protocol in [PROTOCOL.md](PROTOCOL.md).

## Task
Issue 18:00 America/Chicago on day D; predict ERCOT system load (MW, hourly average) for every real
hour of local day D+1 (23/24/25 intervals). Horizon 7–30 h. Load known only through day D−1 (A0).

## Model
Online-weighted ensemble (weights from each member's recent relative error, days ≤ T−2, per
night/day/evening slot group) of three members:
1. **LightGBM** (global over hours) on relative load lags (÷ 7-day level), calendar/holidays,
   11-city GFS temperature forecasts + weighted mean + day aggregates + 1–6 h temperature lags
   (recency effect), observed temperature on T−2/T−7; plus a per-hour online bias correction
   (EWMA half-life 14 days).
2. **Kalman-filtered per-hour Ridge**: hinge (piecewise-linear) temperature terms; coefficients
   follow a random walk (q = 3e-5) updated daily with actuals available at the issue.
3. **Chronos-2** (zero-shot time-series foundation model, amazon/chronos-2, no fine-tuning) with
   2048 h context ending at T−2 and known-future covariates (temperature, hour, day of week, holiday).

Weather at prediction time: mean of GFS and ECMWF IFS 0.25° forecast vintages from runs published
before the issue (GFS +6 h, ECMWF +8 h assumed publication delay); members are trained on GFS.
Fallback when no archived forecast exists: weather-free LightGBM.

## Results (MAE in MW; lower is better)

| Model | Validation 2024–2025 | Holdout 2026-01..09 |
|---|---|---|
| Seasonal naive (T−7) | 4615 | 4286 |
| Ridge, calendar + load lags | 2819 | 2755 |
| LightGBM, no weather | 2854 | 2887 |
| LightGBM + weather + online bias (W8) | 1578 | 1292 |
| Kalman ridge (K3) | 1617 | 1472 |
| 2-member equal ensemble (E2) | 1466 | 1255 |
| **E9 (this model)** | **1310 (MAPE 2.39%)** | **1112 (MAPE 1.86%)** |

Holdout paired block bootstrap (7-day blocks): E9 − E2 = −143 MW, 95% CI [−229, −75];
E9 − W8 = −181 MW, CI [−246, −120]; E9 better in all 3 holdout folds for each comparison.
Observed-weather backcast reference (not a forecast): LightGBM 1144 MW on validation.

External reference, **different protocol** (not comparable as a ranking): ERCOT's own published
Day-Ahead RUC forecast MAPE was about 1.4–3.5% per month in 2024–2025; ERCOT uses same-day load,
proprietary weather inputs and a different issue time.

## What helped and what did not (validation evidence)
- Weather forecasts: 2854 → 1658 (largest single gain).
- Online bias / Kalman adaptation: removed a +600 MW drift bias; modest MAE gains.
- Ensembling diverse members: LGBM + Kalman ridge 1466 vs 1578/1617 alone; adding Chronos-2:
  −73 MW (CI [−101, −46]) although Chronos-2 alone was weaker (1845).
- GFS+ECMWF blend: −54 MW (CI [−86, −19]).
- No significant gain: DAM prices (−8), zonal bottom-up (−17, CI includes 0), extra weather
  variables trained on ERA5 (+7), 4th zonal ensemble member (+3).
- Worse: training on ERA5 and predicting with GFS without correction (bias +1121 MW); Chronos-2
  with 8192 h context (+44, CI [+2, +92]); faster bias adaptation (half-life 3 days).
- Exploratory only: Chronos-2 LoRA fine-tuning on CPU (50 steps, 22 min) did not beat zero-shot on
  a 7-day smoke test; a real study needs a GPU.

## Study S2 (after the holdout was opened): GPU fine-tuning of Chronos-2
Separately labeled study under PROTOCOL.md amendment A4. **E9 above remains the locked S1 model.**
Settings were chosen on a training-period pilot (2023-07..12): zero-shot 1657 → LoRA (1000 steps,
lr 1e-5) 1340, full fine-tune (1000 steps, lr 5e-6) 1338. Chronos-2 is fine-tuned at each fold start
on ERCOT total + 8 zone series (2021-04 to S−2) with GFS temperature and calendar covariates
(RTX A2000 12 GB, ~7 min per fine-tune when run alone).

| Model (validation 2024–2025) | MAE | Paired difference, 95% CI |
|---|---|---|
| T3 Chronos-2 zero-shot | 1845 | |
| T5 Chronos-2 LoRA fine-tuned | 1393 | vs T3 −452 [−572, −342]; vs W8 −185 [−263, −117] |
| T6 Chronos-2 full fine-tuned | 1408 | vs T3 −437 [−559, −331] |
| E9 (S1 main) | 1310 | |
| **E11 = E9 with T5 instead of T3** | **1247 (MAPE 2.27%)** | vs E9 −62 [−89, −39], 7/8 folds |
| E12 = E9 with T6 | 1250 | vs E11 +3 [−3, +8] → simpler E11 kept |

| T8 = T5 with the eval-mode fix (A5) | 1344 | vs T5 −49 [−71, −28], 8/8 folds |
| T7 = eval-mode, 3-seed average | 1325 | vs T8 −19 [−33, −4]; vs W8 −253 [−332, −178] |
| E13 = E9 with T7 | 1247 | vs E11 −0.6 [−9, +9] → E11 kept (pre-registered rule) |

- **Bug (amendment A5):** `Chronos2Pipeline.fit` returns the model in train mode, so T5, T6, E11 and
  E12 forecast with dropout active. Fixing it improved the standalone model by 49 MW; it did not
  change the ensemble, which already averages member noise.
- Fine-tuning made the foundation model the best single model (T7 1325, close to the whole E9
  ensemble), but the ensemble is saturated: as Chronos-2 improved, the correlation of its errors with
  W8 rose from 0.57 (zero-shot) to 0.70, so it adds less diversity. Equal-weight W8+K3+T5 and
  W8+K3+T7 both give 1297.
- S2 candidate (locked in `configs/final_s2.yaml`): **E11**. Caveat: its Chronos member is stochastic
  (dropout on); E13 is the deterministic alternative and is in the S2 holdout set.
- **No holdout result for S2 yet.** 2026-01..09 has been seen; the S2 holdout is target days from
  2026-10-01, run once (`backtest --holdout-s2`, needs `configs/final_s2.yaml`) after ERCOT publishes
  at least 28 days.

## Known failure modes
- **Missing weather vintages**: the GFS archive lacks 2023-12-30..2024-01-19, exactly the Jan-2024
  cold snap; the fallback model's average absolute error on 2024-01-15 was ~23 GW (actual mean load 72 GW). An operational system must
  have redundant weather feeds.
- **Outage-driven events** (Hurricane Beryl 2024-07-08; May-2024 Houston derecho): load drops from
  grid damage are not predictable from weather/load inputs.
- **Weather forecast misses** in cold events: 2025-02-19 had a 3.8 °C error in the forecast
  weighted temperature and a 6.2 GW average under-forecast.
- The worst 5% of validation days carry ~20% of total absolute error.
- Errors grow with horizon (≈1.0 GW at ≤12 h to ≈1.5 GW at 18–30 h) and peak in late afternoon.
- Price-responsive / large flexible loads are not modeled explicitly.

## Limitations of the evaluation
- Load availability (A0) is an assumption: the archive has no publication timestamps; archived
  values may differ from what was visible in real time.
- Weather vintage semantics were inferred empirically (reports/weather_semantics.md); a residual
  uncertainty remains for GFS day-1 at 12–22Z.
- Holiday calendar, city weights and ERA5 proxy for past observed weather are design choices.
- 41 validation runs were used for selection; the holdout was opened once afterwards.

## Reproduce
See README.md. Key commands: `python -m src.elf.backtest --holdout <exp>` (requires configs/final.yaml),
`python -m src.elf.compare A B [--holdout]`, `python -m src.elf.sql_check`.
