"""Paired comparison of two runs on the identical evaluation sample.
Usage: python -m src.elf.compare EXP_A EXP_B [--holdout]"""
import glob, sys
import pandas as pd
from .metrics import block_bootstrap_diff

GAP = (pd.Timestamp("2023-12-30"), pd.Timestamp("2024-01-19"))


def latest(exp, holdout=False):
    runs = [r for r in sorted(glob.glob(f"experiments/runs/*_{exp}{'_HOLDOUT' if holdout else ''}"))]
    runs = [r for r in runs if holdout or not r.endswith("_HOLDOUT")]
    return pd.read_parquet(runs[-1] + "/predictions.parquet")


def compare(a, b, holdout=False):
    A, B = latest(a, holdout), latest(b, holdout)
    m = A.merge(B[["interval_start_utc", "pred"]], on="interval_start_utc", suffixes=("_a", "_b"))
    assert len(m) == len(A) == len(B), "samples differ"
    res = block_bootstrap_diff(m, "pred_a", "pred_b")
    ng = m[~m["T"].between(*GAP)]
    res_ng = block_bootstrap_diff(ng, "pred_a", "pred_b")
    fa = m.groupby("fold").apply(lambda g: (g.pred_a - g.y).abs().mean() < (g.pred_b - g.y).abs().mean()).sum()
    return {"a": a, "b": b, "mae_a": float((m.pred_a - m.y).abs().mean()), "mae_b": float((m.pred_b - m.y).abs().mean()),
            "diff_a_minus_b": res["diff_mae"], "ci95": res["ci95"], "folds_a_wins": int(fa), "folds": int(m.fold.nunique()),
            "excluding_gfs_gap": {"diff": res_ng["diff_mae"], "ci95": res_ng["ci95"]}}


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    print(compare(args[0], args[1], "--holdout" in sys.argv))
