"""Rolling-origin backtest + experiment registry (PROTOCOL.md).

Usage: python -m src.elf.backtest EXP [EXP ...]           # validation folds 2024-2025
       python -m src.elf.backtest --holdout EXP [...]     # only after configs/final.yaml is locked
       python -m src.elf.backtest --holdout-s2 EXP [...]  # study S2 (PROTOCOL A4): target days from
                                                          # 2026-10-01; needs configs/final_s2.yaml, >= 28 new days,
                                                          # and runs once per experiment
"""
import argparse, datetime as dt, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from . import features as F
from .metrics import summarize

VAL_FOLDS = [str(d.date()) for d in pd.date_range("2024-01-01", "2025-10-01", freq="QS")]
VAL_END = "2025-12-31"
HOLDOUT_FOLDS = ["2026-01-01", "2026-04-01", "2026-07-01"]
HOLDOUT_END = "2026-09-30"
S2_HOLDOUT_START = "2026-10-01"
S2_MIN_DAYS = 28
FEAT_CACHE = Path("data/processed/features_A0.parquet")
REG = Path("experiments/registry.jsonl")


def load_features(rebuild=False):
    if FEAT_CACHE.exists() and not rebuild:
        return pd.read_parquet(FEAT_CACHE)
    load = pd.read_parquet("data/processed/load_hourly.parquet")
    df = F.build(load)
    df.to_parquet(FEAT_CACHE, index=False)
    return df


def data_hash():
    return hashlib.sha256(Path("data/processed/load_hourly.parquet").read_bytes()).hexdigest()[:16]


def git_commit():
    try:
        dirty = subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True).stdout.strip()
        c = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        return c + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def make_folds(starts, end):
    out = []
    for i, s in enumerate(starts):
        e = pd.Timestamp(starts[i + 1]) - pd.Timedelta(days=1) if i + 1 < len(starts) else pd.Timestamp(end)
        out.append((pd.Timestamp(s), e))
    return out


def run_backtest(df, model_factory, folds, train_start="2017-02-01", extra_cols=None):
    preds = []
    load_daily = df.groupby("T")["y"].apply(list)
    for i, (s, e) in enumerate(folds):
        tr = df[(df["T"] >= pd.Timestamp(train_start)) & (df["T"] <= s - pd.Timedelta(days=2))]
        te = df[(df["T"] >= s) & (df["T"] <= e)].copy()
        # cutoff check: every training target was observed by the first issue of the fold
        assert tr["T"].max() <= s - pd.Timedelta(days=2)
        assert (te["src_max_day"] <= te["T"] - pd.Timedelta(days=2)).all()
        m = model_factory()
        m.fit(tr)
        te["pred"] = m.predict(te)
        # high-load threshold: q90 of actuals in the 365 days before fold start
        hist = df[(df["T"] < s - pd.Timedelta(days=1)) & (df["T"] >= s - pd.Timedelta(days=366))]["y"]
        te["hl_threshold"] = hist.quantile(0.9)
        te["fold"] = str(s.date())
        preds.append(te)
    return pd.concat(preds, ignore_index=True)


def main(argv=None):
    from .experiments import EXPERIMENTS
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--holdout-s2", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args(argv)
    if a.holdout and not Path("configs/final.yaml").exists():
        sys.exit("Holdout is locked: create configs/final.yaml first (PROTOCOL.md).")
    df = load_features(a.rebuild)
    phase = "holdout" if a.holdout else "validation"
    folds = make_folds(HOLDOUT_FOLDS, HOLDOUT_END) if a.holdout else make_folds(VAL_FOLDS, VAL_END)
    if a.holdout_s2:
        import yaml
        cfg = Path("configs/final_s2.yaml")
        if not cfg.exists():
            sys.exit("S2 holdout is locked: create configs/final_s2.yaml first (PROTOCOL.md A4).")
        allowed = yaml.safe_load(cfg.read_text())["holdout_set"]
        done = {json.loads(l)["experiment"] for l in REG.open() if json.loads(l).get("phase") == "holdout_s2"}
        bad = [e for e in a.exps if e not in allowed or e in done]
        if bad:
            sys.exit(f"Not allowed on the S2 holdout (not in final_s2.yaml or already run once): {bad}")
        last = df["T"].max()
        n_days = (last - pd.Timestamp(S2_HOLDOUT_START)).days + 1
        if n_days < S2_MIN_DAYS:
            sys.exit(f"S2 holdout needs >= {S2_MIN_DAYS} target days after {S2_HOLDOUT_START}; data has {max(n_days, 0)}.")
        phase, folds = "holdout_s2", make_folds([S2_HOLDOUT_START], str(last.date()))
    for name in a.exps:
        spec = EXPERIMENTS[name]
        t0 = time.time()
        status, metrics = "ok", None
        run_id = f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}_{name}{ {'holdout': '_HOLDOUT', 'holdout_s2': '_HOLDOUT_S2'}.get(phase, '')}"
        rdir = Path("experiments/runs") / run_id
        rdir.mkdir(parents=True, exist_ok=True)
        try:
            data = spec["data"](df) if "data" in spec else df
            out = run_backtest(data, spec["model"], folds, spec.get("train_start", "2017-02-01"))
            out[["T", "interval_start_utc", "issue_utc", "horizon_h", "slot", "fold", "y", "pred", "hl_threshold"]].to_parquet(rdir / "predictions.parquet", index=False)
            metrics = summarize(out)
        except Exception as ex:  # failures are logged, not hidden
            status = f"error: {ex!r}"
            import traceback; traceback.print_exc()
        rec = {"run_id": run_id, "experiment": name, "phase": phase,
               "hypothesis": spec["hypothesis"], "config": spec.get("config", {}), "status": status,
               "git": git_commit(), "data_hash": data_hash(), "seconds": round(time.time() - t0, 1),
               "metrics": metrics}
        (rdir / "record.json").write_text(json.dumps(rec, indent=1))
        with REG.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        m = metrics or {}
        print(f"{name:28s} {status:6s} MAE={m.get('mae', float('nan')):8.1f} RMSE={m.get('rmse', float('nan')):8.1f} "
              f"MAPE={m.get('mape', float('nan')):5.2f}% bias={m.get('bias', float('nan')):7.1f} "
              f"peakMAE={m.get('peak_mag_mae', float('nan')):7.1f} HL-MAE={m.get('high_load_mae') or float('nan'):7.1f} "
              f"({rec['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
