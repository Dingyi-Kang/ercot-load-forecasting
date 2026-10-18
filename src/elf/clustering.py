"""Module 2: daily load-profile clustering (descriptive, retrospective).

Rules (PROTOCOL spirit): fit scalers/KMeans and choose k on FIT years only (2017-2023); 2024-2025
are assigned with the frozen model to check stability/interpretability; 2026 is left untouched.
DST days (23/25 h) are excluded from clustering (reported). Cluster labels of a target day use that
day's realized profile, so they are NEVER used as forecast inputs - only for after-the-fact analysis.
Two representations: raw MW (magnitude + shape) and daily-mean-normalized (shape only).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from .features import holiday_table

FIT = ("2017-01-01", "2023-12-31")
CHECK = ("2024-01-01", "2025-12-31")
KS = range(2, 9)
SEEDS = range(10)
FIG = Path("reports/figures")


def daily_profiles():
    load = pd.read_parquet("data/processed/load_hourly.parquet")
    n = load.groupby("op_date").size()
    keep = n[n == 24].index
    excluded = n[n != 24]
    P = load[load.op_date.isin(keep)].pivot(index="op_date", columns="he", values="ERCOT")
    P.index = pd.to_datetime(P.index)
    return P, excluded


def represent(P, kind):
    return P.div(P.mean(axis=1), axis=0) if kind == "shape" else P


def stability(X, k, seeds=SEEDS):
    labs = [KMeans(k, n_init=10, random_state=s).fit_predict(X) for s in seeds]
    aris = [adjusted_rand_score(labs[0], l) for l in labs[1:]]
    return float(np.mean(aris)), float(np.min(aris))


def bootstrap_stability(X, k, reps=20, seed=0):
    """Fit on bootstrap resamples, compare assignments of the full set to the reference fit (ARI)."""
    rng = np.random.default_rng(seed)
    ref = KMeans(k, n_init=10, random_state=0).fit(X)
    out = []
    for _ in range(reps):
        idx = rng.integers(0, len(X), len(X))
        m = KMeans(k, n_init=10, random_state=0).fit(X[idx])
        out.append(adjusted_rand_score(ref.predict(X), m.predict(X)))
    return float(np.mean(out)), float(np.percentile(out, 5))


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    P, excluded = daily_profiles()
    fit = P.loc[FIT[0]:FIT[1]]; chk = P.loc[CHECK[0]:CHECK[1]]
    hol = holiday_table("2016-12-01", "2027-01-01"); hol.index = pd.to_datetime(hol.index)
    report = {"excluded_days": {"n": int(len(excluded)), "lengths": excluded.value_counts().to_dict()},
              "fit_days": int(len(fit)), "check_days": int(len(chk)), "selection": {}}
    lines = ["# Module 2: daily load-profile clustering", "",
             f"Fit years {FIT[0]}..{FIT[1]} ({len(fit)} days); check years {CHECK[0]}..{CHECK[1]} ({len(chk)} days); 2026 untouched.",
             f"Excluded {len(excluded)} DST/incomplete days from clustering (row counts {sorted(set(excluded.values))}).",
             "Labels describe realized days: they are not forecast inputs.", ""]
    models = {}
    for kind in ("raw", "shape"):
        sc = StandardScaler().fit(represent(fit, kind).values)          # scaler fit on FIT years only
        Xf = sc.transform(represent(fit, kind).values)
        Xc = sc.transform(represent(chk, kind).values)
        rows = []
        for k in KS:
            km = KMeans(k, n_init=10, random_state=0).fit(Xf)
            s_fit = silhouette_score(Xf, km.labels_, sample_size=2000, random_state=0)
            s_chk = silhouette_score(Xc, km.predict(Xc)) if len(set(km.predict(Xc))) > 1 else np.nan
            mean_ari, min_ari = stability(Xf, k)
            bmean, b5 = bootstrap_stability(Xf, k)
            sizes = np.bincount(km.labels_, minlength=k) / len(Xf)
            rows.append(dict(k=k, silhouette_fit=s_fit, silhouette_check=s_chk, seed_ari_mean=mean_ari, seed_ari_min=min_ari,
                             boot_ari_mean=bmean, boot_ari_p5=b5, min_cluster_share=sizes.min()))
        tab = pd.DataFrame(rows)
        # pre-stated rule: among k with bootstrap ARI p5 >= 0.8 and min cluster share >= 5%,
        # take the highest fit-silhouette; ties -> smaller k
        ok = tab[(tab.boot_ari_p5 >= 0.8) & (tab.min_cluster_share >= 0.05)]
        k_sel = int(ok.sort_values(["silhouette_fit", "k"], ascending=[False, True]).iloc[0].k) if len(ok) else int(tab.sort_values("boot_ari_mean").iloc[-1].k)
        report["selection"][kind] = {"k": k_sel, "table": tab.round(3).to_dict(orient="records")}
        km = KMeans(k_sel, n_init=10, random_state=0).fit(Xf)
        models[kind] = (sc, km)
        lines += [f"## Representation: {kind}", "", tab.round(3).to_markdown(index=False), "",
                  f"Selected k = **{k_sel}** (rule: bootstrap ARI 5th pct >= 0.8, smallest cluster >= 5%, then highest fit silhouette).", ""]
        # describe clusters on fit + check years
        allP = pd.concat([fit, chk])
        lab = pd.Series(km.predict(sc.transform(represent(allP, kind).values)), index=allP.index)
        desc = pd.DataFrame({"label": lab, "mean_mw": allP.mean(axis=1), "peak_mw": allP.max(axis=1),
                             "peak_he": allP.idxmax(axis=1), "dow": allP.index.dayofweek, "month": allP.index.month,
                             "holiday": hol["holiday"].reindex(allP.index).values != "none",
                             "period": np.where(allP.index <= FIT[1], "fit", "check")})
        g = desc.groupby("label").agg(days=("mean_mw", "size"), mean_mw=("mean_mw", "mean"), peak_mw=("peak_mw", "mean"),
                                      peak_he_median=("peak_he", "median"), weekend_share=("dow", lambda x: (x >= 5).mean()),
                                      holiday_share=("holiday", "mean"))
        months = desc.groupby("label").month.agg(lambda x: ",".join(str(m) for m in x.value_counts().index[:3]))
        g["top_months"] = months
        share = desc.groupby(["label", "period"]).size().unstack(fill_value=0)
        share = share.div(share.sum(axis=0), axis=1)
        g["share_fit"] = share.get("fit"); g["share_check"] = share.get("check")
        # representative day = closest to centroid within fit years
        Xall = sc.transform(represent(allP, kind).values)
        dist = np.linalg.norm(Xall - km.cluster_centers_[lab.values], axis=1)
        rep = pd.Series(dist, index=allP.index).groupby(lab.values).idxmin()
        g["representative_day"] = [str(rep[i].date()) for i in g.index]
        lines += ["Cluster summary (fit + check years; shares per period):", "", g.round(3).to_markdown(), ""]
        lab.to_frame("cluster").assign(representation=kind).to_csv(f"reports/cluster_assignments_{kind}.csv")
        # figure: centroids in original units (inverse scaler)
        C = sc.inverse_transform(km.cluster_centers_)
        plt.figure(figsize=(7, 4))
        for i, c in enumerate(C):
            plt.plot(range(1, 25), c, label=f"cluster {i} (n={int((lab[lab.index <= FIT[1]] == i).sum())})")
        plt.xlabel("hour ending (local)"); plt.ylabel("MW" if kind == "raw" else "load / daily mean")
        plt.title(f"Centroids, {kind} representation (fit 2017-2023)"); plt.legend(fontsize=7); plt.tight_layout()
        plt.savefig(FIG / f"centroids_{kind}.png", dpi=120); plt.close()
        lines += [f"![centroids {kind}](figures/centroids_{kind}.png)", ""]
    # cross-representation agreement
    allP = pd.concat([fit, chk])
    la = models["raw"][1].predict(models["raw"][0].transform(represent(allP, "raw").values))
    lb = models["shape"][1].predict(models["shape"][0].transform(represent(allP, "shape").values))
    lines += [f"ARI between raw and shape clusterings: {adjusted_rand_score(la, lb):.3f} (low = they capture different information).", ""]
    Path("reports/clustering.md").write_text("\n".join(lines) + "\n")
    Path("reports/clustering_summary.json").write_text(json.dumps(report, indent=1, default=str))
    print("\n".join(lines))




def exploratory_shape_k4():
    """EXPLORATORY (not the rule-selected model): shape k=4 also passes the stability criteria
    (bootstrap ARI p5 0.90); finer groups are easier to interpret. Plus retrospective forecast
    error by realized cluster for the main model E9 (validation 2024-2025) - after-the-fact only."""
    import glob
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    P, _ = daily_profiles()
    fit = P.loc[FIT[0]:FIT[1]]; allP = P.loc[FIT[0]:CHECK[1]]
    sc = StandardScaler().fit(represent(fit, "shape").values)
    km = KMeans(4, n_init=10, random_state=0).fit(sc.transform(represent(fit, "shape").values))
    lab = pd.Series(km.predict(sc.transform(represent(allP, "shape").values)), index=allP.index)
    hol = holiday_table("2016-12-01", "2027-01-01"); hol.index = pd.to_datetime(hol.index)
    d = pd.DataFrame({"cluster": lab, "mean_mw": allP.mean(axis=1), "peak_he": allP.idxmax(axis=1),
                      "min_he": allP.idxmin(axis=1), "ratio_peak_mean": allP.max(axis=1) / allP.mean(axis=1),
                      "weekend": allP.index.dayofweek >= 5, "holiday": hol["holiday"].reindex(allP.index).values != "none",
                      "month": allP.index.month})
    g = d.groupby("cluster").agg(days=("mean_mw", "size"), peak_he_median=("peak_he", "median"), min_he_median=("min_he", "median"),
                                 peak_to_mean=("ratio_peak_mean", "mean"), weekend_share=("weekend", "mean"), holiday_share=("holiday", "mean"))
    g["top_months"] = d.groupby("cluster").month.agg(lambda x: ",".join(str(m) for m in x.value_counts().index[:4]))
    run = sorted(r for r in glob.glob("experiments/runs/*_E9_onlineens_lgb14_k3_chronosblend") if not r.endswith("HOLDOUT"))[-1]
    p = pd.read_parquet(run + "/predictions.parquet")
    p["cluster"] = lab.reindex(pd.to_datetime(p["T"])).values
    err = p.dropna(subset=["cluster"]).groupby("cluster").apply(lambda x: pd.Series({
        "val_days": x["T"].nunique(), "E9_mae": (x.pred - x.y).abs().mean(), "E9_bias": (x.pred - x.y).mean(),
        "E9_mape_pct": ((x.pred - x.y).abs() / x.y).mean() * 100}))
    C = sc.inverse_transform(km.cluster_centers_)
    plt.figure(figsize=(7, 4))
    for i, c in enumerate(C):
        plt.plot(range(1, 25), c, label=f"shape cluster {i}")
    plt.xlabel("hour ending (local)"); plt.ylabel("load / daily mean"); plt.title("EXPLORATORY shape k=4 centroids (fit 2017-2023)")
    plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(FIG / "centroids_shape_k4.png", dpi=120); plt.close()
    lines = ["", "## Exploratory view: shape k=4 (not rule-selected; passes stability criteria)", "",
             g.round(3).to_markdown(), "", "![shape k4](figures/centroids_shape_k4.png)", "",
             "### Retrospective: E9 validation error by realized shape cluster (after-the-fact; labels use the target day's actual profile)", "",
             err.round(1).to_markdown(), ""]
    with open("reports/clustering.md", "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))




if __name__ == "__main__":
    main()
    exploratory_shape_k4()
