# ERCOT day-ahead hourly load forecasting (personal research project)

Status: **complete** (holdout opened once); out-of-sample check on 2026-10+ data pending.
Personal project on public data. Not affiliated with ERCOT or Vistra; not a replica of any
company system.

**Question.** Every day at 18:00 America/Chicago, using only information available at that time,
forecast ERCOT system-wide load (MW) for every real hour of the next local day (23/24/25 hours).

**No public leaderboard exists for this task.** Results are internal-protocol comparisons; see
[PROTOCOL.md](PROTOCOL.md) (frozen before tuning; amendments logged) and
[reports/benchmark_validation.md](reports/benchmark_validation.md) (every run, including failures).

## Data (free, public)
- ERCOT Native Load hourly archives 2017–2026 (`src/elf/ingest.py`; hashes in `reports/ingest_manifest.json`).
- Open-Meteo previous-runs archive of GFS / ECMWF forecasts (issue-time-safe vintages; semantics
  check in `reports/weather_semantics.md`) and ERA5 reanalysis (training proxy / backcast only).
  Open-Meteo data is CC BY 4.0, used non-commercially.
- ERCOT Day-Ahead Market hub/zone prices (NP4-180-ER) for "+DAM" variants.
Raw and processed data are not committed (`.gitignore`); scripts re-download them.

## Layout
```
PROTOCOL.md                 task, availability rules, splits, metrics, budget (frozen)
src/elf/ingest.py           ERCOT xlsx -> hourly UTC table (DST-aware)
src/elf/features.py         A0 features: load lags (<= T-2), calendar, holidays
src/elf/weather*.py         forecast-vintage + observed weather features
src/elf/dam.py              DAM price features
src/elf/models.py           naive, Ridge, LightGBM, Kalman ridge, online bias, ensembles
src/elf/tsfm.py             Chronos-2 zero-shot foundation model
src/elf/backtest.py         rolling-origin backtest, registry, locked holdout
src/elf/experiments.py      every experiment with its pre-registered hypothesis
tests/                      DST, cutoff/leakage, metric hand checks
reports/                    benchmark table, related work, semantics notes
experiments/registry.jsonl  append-only run log
```

## Reproduce
```bash
python3 -m venv .venv && .venv/bin/pip install pandas numpy scikit-learn lightgbm duckdb openpyxl pyarrow pyyaml pytest
# download ERCOT archives into data/raw/ercot_native_load (URLs in reports/ingest_manifest.json)
.venv/bin/python -m src.elf.ingest
.venv/bin/python -m src.elf.weather_download
.venv/bin/python -m pytest -q tests
.venv/bin/python -m src.elf.backtest B0_naive_d7 B3_lgbm_rel E2_ens_dyn
.venv/bin/python -m src.elf.report
```

## Status (2026-10-13)
- Done: protocol, ingestion, 41 logged validation runs, locked final model E9,
  holdout opened once ([MODEL_CARD.md](MODEL_CARD.md)), DuckDB SQL layer reconciled with Python,
  Power BI star-schema export + DAX measure drafts, clustering module
  ([reports/clustering.md](reports/clustering.md)), study guide ([reports/STUDY_GUIDE.md](reports/STUDY_GUIDE.md)).
- Headline: validation MAE 1310 MW (MAPE 2.39%), holdout 2026-01..09 MAE 1112 MW (1.86%), vs
  seasonal naive 4286 MW on the holdout. Internal protocol only; no public ranking.
- Study S2 (GPU fine-tuning): validation candidate E11 1247 MW vs E9 1310 (CI [-89, -39]); S2 holdout = 2026-10+ data, not yet available ([MODEL_CARD.md](MODEL_CARD.md)).
- Power BI: data export for a 3-page report covering system load, zone quantiles (Project D) and
  4CP peak risk (Project E). The DAX measures with Python reconciliation targets are in
  [powerbi/MEASURES.md](powerbi/MEASURES.md), and the steps are in [powerbi/BUILD_GUIDE.md](powerbi/BUILD_GUIDE.md).
  The report itself is not built yet.
- Not done: any Snowflake
  work (local DuckDB is not Snowflake), regime-aware model.
