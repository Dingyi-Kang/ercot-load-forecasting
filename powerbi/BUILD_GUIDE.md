# Power BI report — build guide

Goal: a 3-page report in Power BI Desktop (free, Windows) on the forecast outputs of three projects,
with every number reconciled against Python. About 1.5-2 hours the first time.

## 0. Data
```bash
python -m src.elf.export_bi B0_naive_d7 B1_ridge_abs B3_lgbm_rel W8_lgbm_dyn_bias14 \
    K3_kalman_dyn_q3e-5 E2_ens_dyn E9_onlineens_lgb14_k3_chronosblend --holdout
python -m src.elf.export_bi_portfolio     # needs the Project D and E repos next to this one
```
Output: `powerbi/data/*.csv` (3 facts, 3 dimensions, reference metrics). From Windows the folder is
`\\wsl$\<distro>\home\<user>\ML_forecast\powerbi\data` (or copy it to a Windows folder first).

## 1. Load (Power Query)
1. Home > Get data > Text/CSV, load each of: `fact_forecast`, `fact_zone_quantile`,
   `fact_peak_risk`, `dim_date`, `dim_model`, `dim_zone`. Choose **Transform data**, not Load.
2. Set types: every `target_date` -> Date; `local_start` -> Date/Time; `y`, `pred`, `q*`, `*_mw`,
   `p_*` -> Decimal number; `hour_local`, `in_80`, `alert_*`, `is_*` -> Whole number.
   `interval_start_utc` and `issue_utc` can stay Text.
3. Close & Apply. Check the row counts in Data view: 45,857 / 140,160 / 366 for the three facts.

## 2. Relationships (Model view)
Drag to create, all **many-to-one, single cross-filter direction**:
`fact_forecast[target_date]`, `fact_zone_quantile[target_date]`, `fact_peak_risk[target_date]` -> `dim_date[target_date]`;
`fact_forecast[model]`, `fact_zone_quantile[model]` -> `dim_model[model]`;
`fact_zone_quantile[zone]` -> `dim_zone[zone]`.
Delete any relationship Power BI auto-detected between two fact tables.
Mark `dim_date` as date table (Table tools > Mark as date table > `target_date`).
Sort `dim_date[month_name]` by `month`, and `dim_date[dow]` by `dow_num` (Column tools > Sort by column).

## 3. Measures
Create table `_Measures` and add all measures from [MEASURES.md](MEASURES.md), including the
`Miss cost ratio` what-if parameter.

## 4. Pages
**Page 1 — System load forecast (2026 Jan-Sep holdout)**
- Slicers: `dim_model[model_label]` (filter visual to Project 1 models), `dim_date[month_name]`.
- Cards: `MAE (MW)`, `MAPE (%)`, `Bias (MW)`, `MAE vs naive (%)`.
- Line chart: X `fact_forecast[local_start]`, Y `Actual load (MW)` and `Forecast load (MW)`;
  select one model and one week (date range slicer on `dim_date[target_date]`).
- Clustered bar: `dim_model[model_label]` vs `MAE (MW)`, sorted descending.
- Column chart: `fact_forecast[hour_local]` vs `MAE (MW)`, legend = model (pick E9 and LightGBM + weather).

**Page 2 — Zone quantile forecasts (2025 test)**
- Slicers: `dim_model[model_label]` (Project D models), `dim_zone[zone]`.
- Matrix: rows `dim_zone[zone]` + `main_city`, columns `dim_model[model_label]`, values
  `Zone WAPE (%)`, `Coverage 80 (%)`; conditional formatting on `Coverage gap (pp)`.
- Line chart (one zone, one week): X `local_start`, Y `P10 (MW)`, `Median (MW)`, `P90 (MW)`,
  `Zone actual (MW)` — style P10/P90 as dashed lines.
- Column chart: zone vs `Coverage 80 (%)`, analytics pane > constant line at 80.

**Page 3 — 4CP peak risk (summers 2024-2026)**
- Slicers: `dim_date[year]`, `dim_date[month_name]`; the `Miss cost ratio` slider.
- Line chart: X `target_date`, Y sum of `peak_mw`, `forecast_peak_mw`, `month_to_date_max_mw`.
- Column chart: X `target_date`, Y sum of `p_new_month_max`, constant line at 0.40 (the threshold).
- Table: `target_date`, `peak_mw`, `p_new_month_max`, `p_above_bar`, `alert_R1`, `alert_R2`,
  filtered to `is_monthly_peak = 1` (the 12 days that mattered).
- Cards or a small table: `Alerts R0/R1/R2`, `Caught R0/R1/R2`, `Cost R0/R1/R2`, `Brier score`.
  Move the slider from 10 to 20 and watch R2's cost overtake R1 — the main finding of Project E.

**Notes text box on each page:** data source, period, "public data; hourly approximation of 4CP;
costs in alert-day units under an assumed miss/alert cost ratio, no dollar savings".

## 5. Reconcile, then save
Fill the reconciliation table at the end of MEASURES.md. Save as `powerbi/ercot_forecast_report.pbix`
(git-ignored binary) and export File > Export > PDF to `powerbi/ercot_forecast_report.pdf` plus a
screenshot per page into `powerbi/screenshots/` for the README.
