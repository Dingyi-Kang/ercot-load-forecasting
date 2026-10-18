"""Run the DuckDB SQL layer and reconcile it with the Python pipeline."""
import glob, sys
import duckdb
import numpy as np
import pandas as pd
from .backtest import load_features
from .metrics import point_metrics


def main(run_glob="experiments/runs/*_E2_ens_dyn"):
    con = duckdb.connect()
    con.execute("CREATE TABLE load_hourly AS SELECT * FROM 'data/processed/load_hourly.parquet'")
    print(con.execute(open("sql/01_quality.sql").read()).df().T.to_string())
    con.execute(open("sql/02_features.sql").read())   # DuckDB runs multi-statement scripts
    s = con.execute("SELECT * FROM sql_features").df()
    p = load_features()
    s["interval_start_utc"] = pd.to_datetime(s["interval_start_utc"], utc=True)
    m = p.merge(s, on="interval_start_utc", suffixes=("", "_sql"))
    assert len(m) == len(p), (len(m), len(p))
    for c in ["lag_d2", "lag_d7", "d2_mean", "d2_max"]:
        diff = np.nanmax(np.abs(m[c] - m[c + "_sql"]))
        print(f"feature {c}: max |python - sql| = {diff:.6f}  (nulls py={m[c].isna().sum()}, sql={m[c+'_sql'].isna().sum()})")
        assert diff < 1e-6
    run = sorted(glob.glob(run_glob))
    run = [r for r in run if "HOLDOUT" not in r][-1]
    preds = pd.read_parquet(f"{run}/predictions.parquet")
    con.register("preds", preds)
    q = [x for x in open("sql/03_evaluation.sql").read().split(";") if "SELECT" in x]
    print(con.execute(q[0]).df().to_string())
    ev = con.execute(q[1]).df()
    print(ev.to_string())
    py = point_metrics(preds)
    tot = ev[ev["fold"].isna()].iloc[0]
    assert abs(tot["mae_mw"] - py["mae"]) < 1e-6 and abs(tot["rmse_mw"] - py["rmse"]) < 1e-6
    print(f"SQL vs Python overall MAE match: {tot['mae_mw']:.4f} == {py['mae']:.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
