# Power BI measures (DAX)

Create a blank table `_Measures` (Home > Enter data > OK) and add every measure there.
Errors use `err_mw = pred - y`: positive bias = over-forecast.

## Model
| Fact table | Grain | Relationships (many-to-one, single direction) |
|---|---|---|
| `fact_forecast` | model x target hour (system load, 2026 holdout) | `target_date` -> `dim_date[target_date]`; `model` -> `dim_model[model]` |
| `fact_zone_quantile` | model x zone x target hour (UTC) (2025 test) | `target_date` -> `dim_date`; `model` -> `dim_model`; `zone` -> `dim_zone[zone]` |
| `fact_peak_risk` | day (summers 2024-2026) | `target_date` -> `dim_date` |

Actuals (`y`) sit on every model row, so never sum `y` across models; always filter to one model
when showing actual load. MW is a rate: energy (MWh) = MW x hours.

## Page 1 — system load forecast (`fact_forecast`)
```DAX
MAE (MW) = AVERAGE ( fact_forecast[abs_err_mw] )
Bias (MW) = AVERAGE ( fact_forecast[err_mw] )
RMSE (MW) = SQRT ( AVERAGEX ( fact_forecast, fact_forecast[err_mw] ^ 2 ) )
MAPE (%) = AVERAGEX ( fact_forecast, DIVIDE ( fact_forecast[abs_err_mw], fact_forecast[y] ) ) * 100
Hours = COUNTROWS ( fact_forecast )
Actual load (MW) = AVERAGE ( fact_forecast[y] )
Forecast load (MW) = AVERAGE ( fact_forecast[pred] )
Actual energy (GWh) = SUMX ( fact_forecast, fact_forecast[y] * fact_forecast[interval_hours] ) / 1000
Naive MAE (MW) =
    CALCULATE ( [MAE (MW)], REMOVEFILTERS ( dim_model ), dim_model[model] = "B0_naive_d7" )
MAE vs naive (%) = ( DIVIDE ( [MAE (MW)], [Naive MAE (MW)] ) - 1 ) * 100
```

## Page 2 — zone quantile forecasts (`fact_zone_quantile`)
```DAX
Zone WAPE (%) = DIVIDE ( SUM ( fact_zone_quantile[abs_err_mw] ), SUM ( fact_zone_quantile[y] ) ) * 100
Zone WAPE, mean of zones (%) = AVERAGEX ( VALUES ( dim_zone[zone] ), [Zone WAPE (%)] )
Coverage 80 (%) = AVERAGE ( fact_zone_quantile[in_80] ) * 100
Coverage gap (pp) = [Coverage 80 (%)] - 80
Zone actual (MW) = AVERAGE ( fact_zone_quantile[y] )
P10 (MW) = AVERAGE ( fact_zone_quantile[q10] )
Median (MW) = AVERAGE ( fact_zone_quantile[q50] )
P90 (MW) = AVERAGE ( fact_zone_quantile[q90] )
```
`Zone WAPE (%)` pooled over several zones weights big zones more; the model card reports the mean of
per-zone WAPE, which is `Zone WAPE, mean of zones (%)`.

## Page 3 — 4CP peak risk (`fact_peak_risk`)
First create a what-if parameter: Modeling > New parameter > Numeric range, name `Miss cost ratio`,
5 to 30, increment 1, default 10 (this creates `'Miss cost ratio'[Miss cost ratio Value]`).
```DAX
Monthly peaks = SUM ( fact_peak_risk[is_monthly_peak] )
Alerts R0 = SUM ( fact_peak_risk[alert_R0] )
Alerts R1 = SUM ( fact_peak_risk[alert_R1] )
Alerts R2 = SUM ( fact_peak_risk[alert_R2] )
Caught R0 = SUMX ( fact_peak_risk, fact_peak_risk[alert_R0] * fact_peak_risk[is_monthly_peak] )
Caught R1 = SUMX ( fact_peak_risk, fact_peak_risk[alert_R1] * fact_peak_risk[is_monthly_peak] )
Caught R2 = SUMX ( fact_peak_risk, fact_peak_risk[alert_R2] * fact_peak_risk[is_monthly_peak] )
Cost R0 = [Alerts R0] + 'Miss cost ratio'[Miss cost ratio Value] * ( [Monthly peaks] - [Caught R0] )
Cost R1 = [Alerts R1] + 'Miss cost ratio'[Miss cost ratio Value] * ( [Monthly peaks] - [Caught R1] )
Cost R2 = [Alerts R2] + 'Miss cost ratio'[Miss cost ratio Value] * ( [Monthly peaks] - [Caught R2] )
Brier score =
    AVERAGEX (
        FILTER ( fact_peak_risk, NOT ISBLANK ( fact_peak_risk[month_to_date_max_mw] ) ),
        ( fact_peak_risk[p_new_month_max]
            - IF ( fact_peak_risk[peak_mw] > fact_peak_risk[month_to_date_max_mw], 1, 0 ) ) ^ 2
    )
Peak forecast MAPE (%) =
    AVERAGEX ( fact_peak_risk,
        DIVIDE ( ABS ( fact_peak_risk[forecast_peak_mw] - fact_peak_risk[peak_mw] ), fact_peak_risk[peak_mw] ) ) * 100
```

## Reconciliation (must match Python before the report is trusted)
| Check (no slicers unless stated) | Expected | Source |
|---|---|---|
| `MAE (MW)`, model = E9 ensemble | 1,111.8 | `data/reference_metrics.csv` |
| `Hours`, one model | 6,551 | same |
| `MAE (MW)`, model = Seasonal naive | 4,285.6 | same |
| `MAPE (%)`, E9 | 1.86 | MODEL_CARD.md |
| `Zone WAPE, mean of zones (%)`, Chronos-2 LoRA | 3.81 | `data/reference_metrics_zone.csv` (mean of rows) |
| `Coverage 80 (%)`, Chronos-2 LoRA / LightGBM | 72.0 / 56.8 | Project D model card |
| `Alerts R1` / `Caught R1` / `Monthly peaks` | 101 / 10 / 12 | `data/reference_metrics_peak.csv` |
| `Cost R1` / `Cost R2` at ratio 10 | 121 / 124 | Project E model card |
| `Cost R2` at ratio 20 | 224 | same |
| `Brier score` | 0.099 | Project E `reports/E_test_info.json` |
| `Peak forecast MAPE (%)` | 2.61 | same |
