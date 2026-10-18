"""Export a star schema for Power BI (CSV).

fact_forecast : one row per (model, issue_utc, interval_start_utc)  -- grain stated explicitly
dim_date      : one row per local target date
dim_model     : one row per model/run
Usage: python -m src.elf.export_bi EXP [EXP ...] [--holdout]
"""
import glob, json, sys
from pathlib import Path
import pandas as pd
from .features import holiday_table

OUT = Path("powerbi/data")


def main(exps, holdout=False):
    OUT.mkdir(parents=True, exist_ok=True)
    facts, models = [], []
    for e in exps:
        runs = sorted(glob.glob(f"experiments/runs/*_{e}{'_HOLDOUT' if holdout else ''}"))
        runs = [r for r in runs if holdout or not r.endswith("_HOLDOUT")]
        r = runs[-1]
        rec = json.loads(Path(r, "record.json").read_text())
        p = pd.read_parquet(Path(r, "predictions.parquet"))
        p["model"] = e
        p["phase"] = rec["phase"]
        facts.append(p)
        models.append({"model": e, "run_id": rec["run_id"], "phase": rec["phase"], "git": rec["git"],
                       "data_hash": rec["data_hash"], "hypothesis": rec["hypothesis"]})
    f = pd.concat(facts, ignore_index=True)
    assert not f.duplicated(["model", "issue_utc", "interval_start_utc"]).any(), "grain violated"
    f["target_date"] = pd.to_datetime(f["T"]).dt.date
    f["local_start"] = pd.to_datetime(f["interval_start_utc"]).dt.tz_convert("America/Chicago").dt.tz_localize(None)
    f["abs_err_mw"] = (f["pred"] - f["y"]).abs()
    f["err_mw"] = f["pred"] - f["y"]
    f["is_high_load"] = (f["y"] >= f["hl_threshold"]).astype(int)
    f["interval_hours"] = 1.0     # every row is a 1-hour interval: energy MWh = MW * 1 h
    cols = ["model", "phase", "fold", "issue_utc", "interval_start_utc", "local_start", "target_date", "slot", "horizon_h",
            "y", "pred", "err_mw", "abs_err_mw", "is_high_load", "interval_hours"]
    f[cols].to_csv(OUT / "fact_forecast.csv", index=False)
    d = pd.DataFrame({"target_date": sorted(f["target_date"].unique())})
    hol = holiday_table(str(d.target_date.min()), str(d.target_date.max()))
    d["holiday"] = hol["holiday"].reindex(d["target_date"]).values
    dt = pd.to_datetime(d["target_date"])
    d["year"], d["month"], d["dow"], d["is_weekend"] = dt.dt.year, dt.dt.month, dt.dt.day_name(), (dt.dt.dayofweek >= 5).astype(int)
    d.to_csv(OUT / "dim_date.csv", index=False)
    pd.DataFrame(models).to_csv(OUT / "dim_model.csv", index=False)
    # reference numbers for reconciliation with Power BI measures
    ref = f.groupby("model").agg(n=("y", "size"), mae=("abs_err_mw", "mean"), bias=("err_mw", "mean")).reset_index()
    ref.to_csv(OUT / "reference_metrics.csv", index=False)
    print(ref.to_string())


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args, "--holdout" in sys.argv)
