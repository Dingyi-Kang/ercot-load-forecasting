# Evaluation protocol — ERCOT day-ahead hourly load (frozen 2026-10-13, before any tuning)

Changes after this date must be appended
under "Amendments" with a reason; the final holdout definition may not change.

There is **no public leaderboard** for this task. Results are claims of the form
"under this fixed internal protocol, model X beats baseline Y by Z (CI ...)".

## Task

| Item | Definition |
|---|---|
| Target | ERCOT system-wide hourly load, column `ERCOT` of ERCOT "Native Load" yearly archives |
| Unit | MW, hourly average over the interval (numerically equals MWh for a 1-hour interval) |
| Timestamp convention | Archive "Hour Ending" in local prevailing time (America/Chicago); stored as UTC interval start; see `src/elf/ingest.py` |
| Issue time | Daily 18:00 America/Chicago on day D |
| Target times | Every real hourly interval of local operating day T = D+1 (23/24/25 intervals on DST days) |
| Horizon | `interval_end_utc - issue_utc`: 7 h (HE01) … 30 h (HE24) on normal days |

## Information availability at issue (cutoff)

- **Load (primary assumption A0, conservative):** at 18:00 on D, hourly actuals are known for all
  operating days ≤ D−1 (i.e. ≤ T−2). Rationale: ERCOT publishes a daily public report of actual
  hourly load by weather zone (NP6-345-CD). The archive gives final values with no publication
  timestamps, so this is an assumption; real-time telemetry may differ from archived values.
  Same-day load (D, up to 17:00) is **not** used under A0. A later sensitivity run may use it,
  labeled A1, never mixed with A0 results.
- **Calendar:** day of week, local hour, month/season, US federal holidays and fixed
  holiday-adjacent days — known in advance.
- **Weather forecast branch (W-fcst):** only numerical weather forecasts whose model run was
  published before 18:00 on D, with the run's availability time recorded. Source verified before
  use; if no reliable vintage archive is available, the branch is not run.
- **Observed-weather backcast (W-obs):** realized weather at target time. Allowed only as a
  separately labeled retrospective analysis. Never reported as a forecast result.
- All targets of the same issue share the same cutoff. Features that would need data after the
  cutoff are excluded, not imputed from the future.

## Data

- ERCOT Native Load archives 2017–2026 (2016 excluded: different timestamp format, nulls).
  File hashes in `reports/ingest_manifest.json`. Raw/processed data are not committed.
- Known anomalies are flagged, not silently deleted: Winter Storm Uri load shed (2021-02-10 to
  2021-02-20); 2021-05-17/18 zone sum ≠ ERCOT total (target column unaffected).

## Splits

- **Development / validation:** target days 2024-01-01 … 2025-12-31, rolling origin with
  8 quarterly folds. At each fold start S the model is refit on targets with target day ≤ S−2
  (all known at the first issue of the fold), starting 2017-01-01 or later, then frozen for the
  quarter. All model selection uses these folds only.
- **Final holdout (locked):** target days 2026-01-01 … 2026-09-30, same quarterly refit procedure
  (refits at 2026-01-01, 04-01, 07-01). Opened **once**, after the final configuration is locked
  in `configs/final.yaml`. Not used for any choice. Holdout numbers are reported for every model
  in the locked comparison table, not only the winner.

## Metrics

Computed over all (issue, target interval) pairs, pooled across folds.

- **Primary:** MAE (MW).
- **Secondary:** RMSE (MW); MAPE (%; load is never ≤ 0 in this data); bias = mean(pred − actual);
  daily-peak magnitude abs error = |max pred − max actual| per target day; daily-peak timing error
  = |hour of predicted max − hour of actual max| (h); high-load MAE on intervals whose actual ≥ the
  90th percentile of actual load over the 365 days before the fold start; MAE by month, by horizon,
  by local hour; per-fold MAE.
- **Uncertainty:** moving-block bootstrap over target days (7-day blocks, 1000 reps) of the MAE
  difference between two models on identical samples; also count of folds won.

## Baselines (implemented before any optimization)

1. Seasonal naive: same local hour on T−7 (available under A0).
2. Seasonal naive with level adjustment variants are optimizations, not baselines.
3. Ridge regression on calendar + available load-history features (train-only scaling).
4. LightGBM on the same features.

## Tuning budget and logging

- At most 60 registered validation runs before locking. Every run, including failures, appends a
  record to `experiments/registry.jsonl` (config, git commit, data hash, metrics, timestamp).
- Every experiment states a hypothesis before running; results are kept even when worse.
- No seed cherry-picking: stochastic models report the configured seed; seed variance is checked
  for the final candidate.

## Compute and data limits

Local CPU only (24 cores, 58 GB RAM). Free public data only. No paid services, no leaderboard
submissions, no uploads.

## Amendments

### A1 (2026-10-13, before any weather/online experiment; baselines B0-B3 already run)
- **Online state updates allowed.** Within a fold, hyperparameters and feature sets stay frozen,
  but a model may update internal state (e.g. Kalman-filtered coefficients, recent-error bias
  correction, online expert weights) at each issue using only actuals available at that issue
  (target days <= T-2). This mirrors the competition winner's practice (de Vilmarest & Goude 2022).
- **Weather forecast vintages:** Open-Meteo previous-runs archive; empirically the `_previous_dayN`
  value at valid time v comes from the run initialised at floor6(v - 24N h) (checked against the
  single-runs archive, see reports/weather_semantics.md). A run is treated as usable 6 h after
  initialisation. GFS temperature exists from 2021-03-25; other variables and ECMWF from 2024.
- **Observed weather for past days:** observed weather for operating days <= T-2 is treated as
  known at issue (station observations are real time; ERA5 reanalysis is used as a proxy).
  Observed weather at target times is backcast-only (W-obs) and reported separately.
- Weather-forecast models are trained from 2021-04-01 at the earliest (vintage availability);
  a model trained on observed weather but fed forecasts at prediction time is a valid forecast
  model and is labeled "train-obs / predict-fcst".

### A2 (2026-10-13, before any DAM experiment; 18 validation runs used so far)
- **ERCOT Day-Ahead Market prices** for operating day T (hub/load-zone settlement point prices,
  NP4-180-ER archives) are treated as known at the 18:00 issue on D: the DAM clears in the morning
  and results are posted early afternoon of D. They aggregate market participants' expectations
  (including their load and weather forecasts), so DAM-based results are labeled "+DAM" and also
  reported without DAM. Days with delayed DAM postings are not identified in the archive (limitation).

### A3 (2026-10-13 03:40 CDT, written before results of T2, W9, E4, W10, E5, Z1 were seen)
**Final selection rule.** After the validation budget is spent or no hypothesis remains:
1. Main model = lowest pooled validation MAE among forecast (non-backcast) runs that use **no DAM**
   inputs. If the gain over the next simpler candidate is inside the block-bootstrap 95% CI, prefer
   the simpler model (fewer components).
2. A "+DAM" variant is reported separately, never as the headline.
3. The holdout (2026-01-01..2026-09-30) is run **once** for this fixed set: B0, B1, B3, W8,
   K3 (best Kalman), E2, the main model, and its +DAM variant if it exists. All numbers reported.
4. No change to any model after the holdout is opened; later ideas become a new, separately
   labeled study with a new holdout.

### A4 (2026-10-13 09:20 CDT, after the A3 holdout was opened) — Study S2: GPU fine-tuning of Chronos-2
A local GPU (NVIDIA RTX A2000 12 GB) became available. Per A3.4 this is a **new, separately
labeled study**; E9 stays the locked main model of study S1 and its holdout numbers are unchanged.
- **Pilot (setting choice, no validation data):** fine-tune on data <= 2023-06-30, evaluate on
  target days 2023-07-01..2023-12-28 (training period, before validation and before the GFS gap).
  Candidates: zero-shot, LoRA and full fine-tuning at a few step counts / learning rates. The
  pilot picks at most two settings; pilot results are logged in experiments/exploratory_notes.txt.
- **Validation:** the chosen settings run on the S1 validation folds (2024-2025) with a fresh
  fine-tune at each fold start on data available then (<= S-2), as standalone members (vs T3)
  and inside the E9 ensemble (replacing T3). Budget: 8 validation runs.
- **Selection:** a fine-tuned variant replaces T3 only if the ensemble's validation MAE is lower
  and the block-bootstrap 95% CI of the difference excludes 0.
- **Holdout for S2:** target days from 2026-10-01 onward, opened once when ERCOT publishes at
  least one full month. 2026-01..09 has been seen and is **not** an S2 holdout; any number on it
  for S2 models is labeled post-hoc.

### A5 (2026-10-14 ~02:30 CDT, S2; before any A5 validation result)
- **Bug found in S2:** `Chronos2Pipeline.fit` returns the fine-tuned model in train mode, so T5, T6,
  E11 and E12 predicted with dropout active (random noise in every forecast). Those runs stay in the
  registry as run (no leakage involved); new runs call `model.eval()` after fine-tuning. Pilot
  (2023-H2): eval mode 1318/1269/1291 for seeds 0/1/2 vs 1344/1385/1368 with dropout on.
- **Seed averaging:** average of 3 LoRA fine-tunes differing only in seed; pilot 1263 vs single
  seeds 1269-1318. Zone temperatures and LoRA rank 16 were within seed noise in the pilot: dropped.
- **Runs (3 of the 4 remaining S2 runs):** T7 (3-seed, eval), T8 (seed 0, eval; isolates the fix),
  E13 (E9 with T7). Fine-tuned weights are cached by (config, seed, training cutoff) to avoid
  recomputing identical fits; this changes compute only.
- **Selection:** E13 replaces E11 as the S2 candidate only if E13 − E11 has a 95% CI excluding 0.
