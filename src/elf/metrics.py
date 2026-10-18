"""Metrics defined in PROTOCOL.md."""
import numpy as np
import pandas as pd


def point_metrics(df, pred="pred", y="y"):
    e = df[pred] - df[y]
    return {"n": int(len(df)), "mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "mape": float((e.abs() / df[y]).mean() * 100), "bias": float(e.mean())}


def peak_metrics(df, pred="pred", y="y"):
    rows = []
    for T, g in df.groupby("T"):
        ia, ip = g[y].values.argmax(), g[pred].values.argmax()
        rows.append((abs(g[pred].values.max() - g[y].values.max()),
                     abs((g["interval_start_utc"].iloc[ip] - g["interval_start_utc"].iloc[ia]).total_seconds()) / 3600))
    a = np.array(rows)
    return {"peak_mag_mae": float(a[:, 0].mean()), "peak_timing_mae_h": float(a[:, 1].mean()),
            "peak_timing_exact_pct": float((a[:, 1] == 0).mean() * 100)}


def summarize(df, pred="pred"):
    out = point_metrics(df, pred)
    out.update(peak_metrics(df, pred))
    hl = df[df["y"] >= df["hl_threshold"]]
    out["high_load_mae"] = float((hl[pred] - hl["y"]).abs().mean()) if len(hl) else None
    out["high_load_n"] = int(len(hl))
    out["by_fold_mae"] = {str(k): float(v) for k, v in df.groupby("fold").apply(lambda g: (g[pred] - g["y"]).abs().mean()).items()}
    out["by_month_mae"] = {int(k): float(v) for k, v in df.groupby(df["T"].dt.month).apply(lambda g: (g[pred] - g["y"]).abs().mean()).items()}
    return out


def block_bootstrap_diff(df, pred_a, pred_b, block=7, reps=1000, seed=0):
    """MAE(a) - MAE(b) with moving-block bootstrap over target days. Negative = a better."""
    d = df.assign(ea=(df[pred_a] - df["y"]).abs(), eb=(df[pred_b] - df["y"]).abs())
    g = d.groupby("T").agg(ea=("ea", "sum"), eb=("eb", "sum"), n=("y", "size")).sort_index()
    ea, eb, n = g.ea.values, g.eb.values, g.n.values
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(len(g) / block))
    stats = []
    for _ in range(reps):
        starts = rng.integers(0, len(g) - block + 1, nb)
        idx = (starts[:, None] + np.arange(block)).ravel()[: len(g)]
        stats.append((ea[idx].sum() - eb[idx].sum()) / n[idx].sum())
    point = (ea.sum() - eb.sum()) / n.sum()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"diff_mae": float(point), "ci95": [float(lo), float(hi)]}
