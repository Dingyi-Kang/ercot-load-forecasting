"""Failure analysis for one run -> reports/error_analysis_<exp>.md"""
import glob, sys
from pathlib import Path
import numpy as np
import pandas as pd

GAP = (pd.Timestamp("2023-12-30"), pd.Timestamp("2024-01-19"))
EVENTS = {  # publicly documented events, used only to label days after the fact
    "2024-01-13..21": "Jan-2024 cold snap (also inside the GFS archive gap -> fallback model)",
    "2024-07-08..10": "Hurricane Beryl landfall near Houston; large outages",
    "2024-05-16..17": "Houston derecho; large outages",
}


def main(exp, holdout=False):
    runs = sorted(glob.glob(f"experiments/runs/*_{exp}{'_HOLDOUT' if holdout else ''}"))
    runs = [r for r in runs if holdout or not r.endswith("_HOLDOUT")]
    p = pd.read_parquet(runs[-1] + "/predictions.parquet")
    w = pd.read_parquet("data/processed/features_A0_weather.parquet", columns=["interval_start_utc", "wf_temp_wavg", "wo_temp_wavg", "wb_temp_wavg"])
    p = p.merge(w, on="interval_start_utc", how="left")
    p["e"] = p.pred - p.y; p["ae"] = p.e.abs()
    p["temp_fc_err"] = (p.wb_temp_wavg - p.wo_temp_wavg).abs()
    L = [f"# Error analysis: {exp} ({'holdout' if holdout else 'validation'})", "", ""]
    L += [f"Overall MAE {p.ae.mean():.0f} MW; excluding GFS-gap days {p[~p['T'].between(*GAP)].ae.mean():.0f} MW; n={len(p)}.", ""]
    m = p.groupby(p["T"].dt.to_period("M")).agg(mae=("ae", "mean"), bias=("e", "mean"), temp_err=("temp_fc_err", "mean"))
    L += ["## By month", "", m.round(1).to_markdown(), ""]
    h = p.groupby("slot").agg(mae=("ae", "mean"), bias=("e", "mean"))
    L += ["## By local hour-ending slot", "", h.round(0).T.to_markdown(), ""]
    hz = p.groupby(pd.cut(p.horizon_h, [0, 12, 18, 24, 31])).ae.mean().round(0)
    L += ["## By horizon (h from issue to interval end)", "", hz.to_frame("mae").to_markdown(), ""]
    d = p.groupby("T").agg(mae=("ae", "mean"), bias=("e", "mean"), actual_mean=("y", "mean"), temp_fc_err=("temp_fc_err", "mean"))
    top = d.sort_values("mae", ascending=False).head(12)
    L += ["## Worst 12 target days", "", top.round(1).to_markdown(), "",
          "Labels (public events, applied after the fact, not model inputs):"] + [f"- {k}: {v}" for k, v in EVENTS.items()]
    share = d.mae.sort_values(ascending=False)
    L += ["", f"Worst 5% of days account for {share.head(int(len(d)*0.05)).sum()/share.sum()*100:.1f}% of total absolute error.", ""]
    c = np.corrcoef(d.temp_fc_err.fillna(d.temp_fc_err.mean()), d.mae)[0, 1]
    L += [f"Correlation between daily temperature-forecast error (|blend - ERA5|, °C) and daily MAE: {c:.2f}.", ""]
    out = Path(f"reports/error_analysis_{exp}{'_holdout' if holdout else ''}.md")
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main(sys.argv[1], "--holdout" in sys.argv)
