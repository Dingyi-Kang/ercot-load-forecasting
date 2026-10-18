"""Export the star schema for the Power BI forecast report (powerbi/BUILD_GUIDE.md).

Facts (grain stated; every fact joins dim_date on target_date, many-to-one):
  fact_forecast       model x target hour            Project 1 system load, 2026 holdout (from export_bi)
  fact_zone_quantile  model x zone x target hour (UTC; local_start repeats on the DST fall-back day)    Project D zone quantiles, 2025 test
  fact_peak_risk      day                            Project E 4CP peak risk, summers 2024-2026 (test)
Dimensions: dim_date (2024-01-01..2026-12-31), dim_model, dim_zone.
reference_metrics_*.csv hold Python numbers that the DAX measures must reproduce.

Usage: python -m src.elf.export_bi_portfolio   (run `python -m src.elf.export_bi ... --holdout` first)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from .features import holiday_table

OUT = Path("powerbi/data")
TZ = "America/Chicago"
D_PREDS = Path.home() / "ML_transformer_load/reports/preds"
E_DAILY = Path.home() / "ML_4cp/reports/E_test_daily.parquet"
D_MODELS = {"D3_chronos2_lora": "Chronos-2 LoRA (selected)", "D1_lgbm_quantile": "LightGBM quantile"}
P1_LABELS = {"B0_naive_d7": "Seasonal naive", "B1_ridge_abs": "Ridge baseline", "B3_lgbm_rel": "LightGBM, no weather",
             "W8_lgbm_dyn_bias14": "LightGBM + weather", "K3_kalman_dyn_q3e-5": "Kalman ridge",
             "E2_ens_dyn": "2-model ensemble", "E9_onlineens_lgb14_k3_chronosblend": "E9 ensemble (selected)"}
ZONES = {"COAST": "Houston", "EAST": "Tyler", "FWEST": "Midland", "NORTH": "Wichita Falls", "NCENT": "Dallas-Fort Worth",
         "SOUTH": "Corpus Christi", "SCENT": "Austin / San Antonio", "WEST": "Abilene"}


def forecast():
    f = pd.read_csv(OUT / "fact_forecast.csv")
    f["hour_local"] = pd.to_datetime(f["local_start"]).dt.hour
    f["model_label"] = f["model"].map(P1_LABELS)
    f.to_csv(OUT / "fact_forecast.csv", index=False)
    return f


def zone_quantile():
    rows = []
    for m, label in D_MODELS.items():
        p = pd.read_parquet(D_PREDS / f"{m}__test.parquet")
        loc = pd.DatetimeIndex(p["interval_start_utc"]).tz_convert(TZ).tz_localize(None)
        rows.append(pd.DataFrame({
            "model": m, "model_label": label, "zone": p["zone"].values,
            "target_date": pd.to_datetime(p["T"]).dt.date.values,
            "interval_start_utc": p["interval_start_utc"].values, "local_start": loc, "hour_local": loc.hour,
            "y": p["y"].values, "q10": p["q10"].values, "q50": p["q50"].values, "q90": p["q90"].values,
            "abs_err_mw": (p["q50"] - p["y"]).abs().values,
            "in_80": ((p["y"] >= p["q10"]) & (p["y"] <= p["q90"])).astype(int).values}))
    z = pd.concat(rows, ignore_index=True)
    assert not z.duplicated(["model", "zone", "interval_start_utc"]).any(), "grain violated"
    z.to_csv(OUT / "fact_zone_quantile.csv", index=False)
    ref = z.groupby(["model", "zone"]).apply(lambda g: pd.Series({
        "wape_pct": 100 * g["abs_err_mw"].sum() / g["y"].sum(), "cov80_pct": 100 * g["in_80"].mean()}),
        include_groups=False).reset_index()
    ref.to_csv(OUT / "reference_metrics_zone.csv", index=False)
    return z


def peak_risk(tau=0.40):
    e = pd.read_parquet(E_DAILY).reset_index().rename(columns={"day": "target_date"})
    first = e["mtd"].isna()
    out = pd.DataFrame({
        "target_date": e["target_date"].dt.date, "peak_mw": e["peak"], "forecast_peak_mw": e["pred"],
        "month_to_date_max_mw": e["mtd"], "expected_bar_mw": e["bar"],
        "p_new_month_max": e["p_mtd"], "p_above_bar": e["p_bar"], "is_monthly_peak": e["is_cp"],
        "alert_R1": (e["p_mtd"] >= tau).astype(int), "alert_R2": (e["p_bar"] >= tau).astype(int),
        "alert_R0": (first | (e["pred"] >= 0.98 * e["mtd"])).astype(int)})
    out.to_csv(OUT / "fact_peak_risk.csv", index=False)
    ref = pd.DataFrame([{"rule": r, "alerts": int(out[f"alert_{r}"].sum()),
                         "peaks_caught": int((out[f"alert_{r}"] * out["is_monthly_peak"]).sum()),
                         "peaks": int(out["is_monthly_peak"].sum())} for r in ("R0", "R1", "R2")])
    ref.to_csv(OUT / "reference_metrics_peak.csv", index=False)
    return out


def dims():
    d = pd.DataFrame({"target_date": pd.date_range("2024-01-01", "2026-12-31", freq="D")})
    hol = holiday_table("2024-01-01", "2026-12-31")
    d["holiday"] = hol["holiday"].reindex(d["target_date"]).fillna("none").values
    d["year"], d["month"] = d.target_date.dt.year, d.target_date.dt.month
    d["month_name"] = d.target_date.dt.strftime("%b")
    d["dow"], d["dow_num"] = d.target_date.dt.day_name(), d.target_date.dt.dayofweek
    d["is_weekend"] = (d["dow_num"] >= 5).astype(int)
    d["season"] = np.select([d.month.isin([12, 1, 2]), d.month.isin([3, 4, 5]), d.month.isin([6, 7, 8])],
                            ["Winter", "Spring", "Summer"], "Fall")
    d["target_date"] = d["target_date"].dt.date
    d.to_csv(OUT / "dim_date.csv", index=False)
    m = pd.read_csv(OUT / "dim_model.csv")
    m = m[~m["model"].isin(list(D_MODELS))]          # idempotent re-runs
    m["model_label"] = m["model"].map(P1_LABELS)
    m["project"] = "System load (Project 1)"
    extra = pd.DataFrame({"model": list(D_MODELS), "model_label": list(D_MODELS.values()),
                          "project": "Zone quantiles (Project D)", "phase": "test"})
    pd.concat([m, extra], ignore_index=True).to_csv(OUT / "dim_model.csv", index=False)
    pd.DataFrame({"zone": list(ZONES), "main_city": list(ZONES.values())}).to_csv(OUT / "dim_zone.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f, z, p = forecast(), zone_quantile(), peak_risk()
    dims()
    dd = set(pd.read_csv(OUT / "dim_date.csv")["target_date"])
    for name, df in (("fact_forecast", f), ("fact_zone_quantile", z), ("fact_peak_risk", p)):
        missing = set(df["target_date"].astype(str)) - dd
        assert not missing, f"{name}: dates missing from dim_date: {sorted(missing)[:3]}"
        print(f"{name}: {len(df):,} rows")
    print(pd.read_csv(OUT / "reference_metrics_peak.csv").to_string(index=False))


if __name__ == "__main__":
    main()
