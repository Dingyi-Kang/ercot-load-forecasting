"""Model zoo. Each model: fit(train_df) / predict(df) -> MW array."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

LAGS = ["lag_d2", "lag_d3", "lag_d4", "lag_d5", "lag_d6", "lag_d7", "lag_d14", "lag_d21", "lag_d28",
        "lag_d2_prev", "lag_d2_next", "lag_d7_prev", "lag_d7_next", "d2_mean", "d2_max", "d2_min", "d2_last",
        "d3_mean", "d9_mean", "level28"]


class SeasonalNaive:
    name = "naive_d7"
    def fit(self, tr): return self
    def predict(self, df): return df["lag_d7"].values


def _calendar_dummies(df, cols):
    parts = []
    if "dow" in cols:
        parts.append(pd.get_dummies(df["dow"], prefix="dow").reindex(columns=[f"dow_{i}" for i in range(7)], fill_value=0))
    if "month" in cols:
        parts.append(pd.get_dummies(df["month"], prefix="m").reindex(columns=[f"m_{i}" for i in range(1, 13)], fill_value=0))
    if "holiday" in cols:
        parts.append(df[["is_holiday", "holiday_d7", "holiday_d2", "holiday_next", "holiday_prev", "xmas_week"]])
    return pd.concat(parts, axis=1).astype(float)


class RidgePerSlot:
    """24 Ridge models (one per local hour slot). Load features are expressed relative to the
    7-day level (level7) so the model transfers across the strong upward trend; the target is
    y/level7 and predictions are scaled back. Scaler is fit on training rows only."""
    def __init__(self, alpha=1.0, relative=True, lags=LAGS, cal=("dow", "month", "holiday")):
        self.alpha, self.relative, self.lags, self.cal = alpha, relative, list(lags), cal
        self.name = f"ridge_slot_a{alpha}{'_rel' if relative else ''}"

    def _X(self, df):
        L = df[self.lags].copy()
        if self.relative:
            L = L.div(df["level7"], axis=0)
        X = pd.concat([L, _calendar_dummies(df, self.cal),
                       df[["doy_sin", "doy_cos"]]], axis=1)
        return X.astype(float)

    def fit(self, tr):
        tr = tr.dropna(subset=self.lags + ["level7", "y"])
        self.models = {}
        for s, g in tr.groupby("slot"):
            X = self._X(g)
            yy = g["y"] / g["level7"] if self.relative else g["y"]
            sc = StandardScaler().fit(X)
            self.models[s] = (sc, Ridge(alpha=self.alpha).fit(sc.transform(X), yy), list(X.columns))
        return self

    def predict(self, df):
        out = np.full(len(df), np.nan)
        for s, idx in df.groupby("slot").indices.items():
            sc, m, cols = self.models[s]
            g = df.iloc[idx]
            p = m.predict(sc.transform(self._X(g)[cols]))
            out[idx] = p * g["level7"].values if self.relative else p
        return out


class LGBM:
    """Global LightGBM over all slots, target relative to level7 by default."""
    def __init__(self, relative=True, params=None, features=None, name="lgbm", seed=0):
        import lightgbm  # noqa
        self.relative = relative
        self.params = dict(n_estimators=1500, learning_rate=0.03, num_leaves=63, min_child_samples=50,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                           objective="l1", verbose=-1, random_state=seed, n_jobs=16)
        self.params.update(params or {})
        self.features = features
        self.name = name

    def _X(self, df):
        cols = self.features or (LAGS + ["slot", "dow", "month", "doy_sin", "doy_cos", "is_holiday", "holiday_d7",
                                         "holiday_d2", "holiday_next", "holiday_prev", "xmas_week", "is_dst", "dst_dup"])
        X = df[[c for c in cols]].copy()
        if self.relative:
            for c in LAGS:
                if c in X:
                    X[c] = X[c] / df["level7"]
        return X.astype(float)

    def fit(self, tr):
        import lightgbm as lgb
        tr = tr.dropna(subset=["level7", "y"])
        yy = tr["y"] / tr["level7"] if self.relative else tr["y"]
        self.m = lgb.LGBMRegressor(**self.params).fit(self._X(tr), yy)
        return self

    def predict(self, df):
        p = self.m.predict(self._X(df))
        return p * df["level7"].values if self.relative else p


# ---------------------------------------------------------------- weather-aware linear models
KNOTS = [0, 5, 10, 15, 20, 25, 28, 31, 34]


def hinge(t, knots=KNOTS):
    """Piecewise-linear temperature basis (GAM-like U shape) for linear models."""
    t = np.asarray(t, float)
    cols = {"t": t}
    for k in knots:
        cols[f"h{k}"] = np.maximum(t - k, 0)
    return pd.DataFrame(cols)


class RidgeWeather(RidgePerSlot):
    """Per-slot Ridge: relative load lags + calendar + hinge(temperature) features.
    Expects weather columns already mapped to w_* by WeatherWrap."""
    def __init__(self, alpha=1.0, extra=("w_dew_wavg",), use_cities=False, name=None,
                 hinge_cols=("w_temp_wavg", "w_tday_mean", "w_tday_max")):
        super().__init__(alpha=alpha, relative=True)
        self.extra, self.use_cities, self.hinge_cols = list(extra), use_cities, list(hinge_cols)
        self.name = name or f"ridgeW_a{alpha}"

    def _X(self, df):
        base = super()._X(df).reset_index(drop=True)
        parts = [base]
        for col in self.hinge_cols:
            pre = col
            h = hinge(df[col].values); h.columns = [f"{pre}_{c}" for c in h.columns]; parts.append(h)
        o2 = pd.DataFrame({"o2_temp": df["o2_temp_wavg"].values, "dT2": (df["w_tday_mean"] - df["o2_tday_mean"]).values,
                           "o7_temp": df["o7_temp_wavg"].values})
        h2 = hinge(df["o2_temp_wavg"].values); h2.columns = [f"o2_{c}" for c in h2.columns]
        parts += [o2, h2]
        ex = [c for c in self.extra if c in df.columns]
        if ex:
            parts.append(df[ex].reset_index(drop=True))
        if self.use_cities:
            parts.append(df[[c for c in df.columns if c.startswith("w_temp_") and c != "w_temp_wavg"]].reset_index(drop=True))
        X = pd.concat(parts, axis=1)
        X.index = df.index
        return X.astype(float)

    def fit(self, tr):
        need = ["w_temp_wavg", "o2_temp_wavg", "o7_temp_wavg"] + [c for c in self.extra if c in tr.columns]
        return super().fit(tr.dropna(subset=need))


class KalmanRidge(RidgeWeather):
    """RidgeWeather whose per-slot coefficients follow a random walk and are updated by a Kalman
    filter at each issue with actuals of day T-2 only (protocol amendment A1).
    q: process-noise scale relative to observation noise (0 => static ridge)."""
    def __init__(self, q=1e-3, p0=1e-2, **kw):
        super().__init__(**kw)
        self.q, self.p0 = q, p0
        self.name = f"kalman_q{q}_p{p0}"

    def fit(self, tr):
        super().fit(tr)
        need = ["w_temp_wavg", "o2_temp_wavg", "o7_temp_wavg", "level7", "y"] + self.lags
        tr = tr.dropna(subset=need)
        self.state = {}
        for s, g in tr.groupby("slot"):
            sc, m, cols = self.models[s]
            X = sc.transform(self._X(g)[cols]); yy = (g["y"] / g["level7"]).values
            r = np.var(yy - m.predict(X)) + 1e-8
            theta = np.r_[m.intercept_, m.coef_]
            p = len(theta)
            self.state[s] = dict(theta=theta, P=np.eye(p) * self.p0 * r, Q=np.eye(p) * self.q * r, R=r)
        self.last_tr_day = tr["T"].max()
        return self

    def _phi(self, s, g):
        sc, m, cols = self.models[s]
        X = sc.transform(self._X(g)[cols])
        return np.c_[np.ones(len(X)), X]

    def predict(self, df):
        """Walk forward day by day. Before predicting day T, assimilate actuals of days up to T-2
        that are in `df` (the evaluation fold) - those actuals are known at T's issue."""
        orig = df.index
        df = df.sort_values(["T", "interval_start_utc"])
        out = pd.Series(np.nan, index=df.index)
        days = sorted(df["T"].unique())
        byday = {d: g for d, g in df.groupby("T")}
        assimilated = set()
        for T in days:
            for d in days:
                if d <= T - pd.Timedelta(days=2) and d not in assimilated:
                    self._update(byday[d]); assimilated.add(d)
            g = byday[T]
            pr = np.empty(len(g))
            for s, idx in g.groupby("slot").indices.items():
                gs = g.iloc[idx]
                pr[idx] = self._phi(s, gs) @ self.state[s]["theta"] * gs["level7"].values
            out.loc[g.index] = pr
        return out.loc[orig].values

    def _update(self, g):
        g = g.dropna(subset=["y", "level7"])
        for s, idx in g.groupby("slot").indices.items():
            gs = g.iloc[idx]
            st = self.state[s]
            for x, yv in zip(self._phi(s, gs), (gs["y"] / gs["level7"]).values):
                P = st["P"] + st["Q"]
                Px = P @ x
                k = Px / (x @ Px + st["R"])
                st["theta"] = st["theta"] + k * (yv - x @ st["theta"])
                st["P"] = P - np.outer(k, Px)


class OnlineBias:
    """Wrap a model with a per-slot online bias correction (amendment A1).
    Correction = EWMA (half-life `halflife` days) of relative residuals (y - pred)/level7 over days
    whose actuals are known at the issue (<= T-2). Warm start: the base model is first fit without
    the last `warm` training days, which give out-of-sample residuals; then refit on all training."""
    def __init__(self, factory, halflife=7.0, warm=28):
        self.factory, self.halflife, self.warm = factory, halflife, warm
        self.name = f"onlinebias_hl{halflife}"

    def fit(self, tr):
        last = tr["T"].max()
        early = tr[tr["T"] <= last - pd.Timedelta(days=self.warm)]
        m0 = self.factory().fit(early)
        w = tr[tr["T"] > last - pd.Timedelta(days=self.warm)].copy()
        w["pred"] = m0.predict(w)
        self.hist = w[["T", "slot", "y", "pred", "level7"]].copy()
        self.m = self.factory().fit(tr)
        return self

    def predict(self, df):
        base = self.m.predict(df)
        d = df[["T", "slot", "y", "level7"]].copy()
        d["pred"] = base
        hist = pd.concat([self.hist, d], ignore_index=True)
        hist["r"] = (hist["y"] - hist["pred"]) / hist["level7"]
        daily = hist.groupby(["T", "slot"])["r"].mean().unstack("slot").sort_index()
        alpha = 1 - 0.5 ** (1 / self.halflife)
        ew = daily.ewm(alpha=alpha, ignore_na=True).mean()
        # correction for target day T uses residuals up to T-2 only
        ew.index = ew.index + pd.Timedelta(days=2)
        corr = ew.reindex(pd.DatetimeIndex(df["T"].unique()).union(ew.index)).ffill()
        c = np.array([corr.at[t, s] if s in corr.columns else 0.0 for t, s in zip(df["T"], df["slot"])])
        c = np.nan_to_num(c)
        return base + c * df["level7"].values


class Ensemble:
    """Fixed-weight average of member models (weights chosen a priori, not fit on validation)."""
    def __init__(self, factories, weights=None, name="ens"):
        self.factories, self.weights, self.name = factories, weights, name

    def fit(self, tr):
        self.ms = [f().fit(tr) for f in self.factories]
        return self

    def predict(self, df):
        P = np.column_stack([m.predict(df) for m in self.ms])
        w = np.ones(P.shape[1]) / P.shape[1] if self.weights is None else np.asarray(self.weights)
        return P @ w


class OnlineEnsemble:
    """Ensemble whose weights follow recent performance (amendment A1): for target day T, member
    weight ∝ exp(-eta * EWMA relative abs error up to day T-2), per slot group (night/day/evening).
    Warm start: members are first fit without the last `warm` training days to obtain out-of-sample
    errors, then refit on all training data (same procedure as OnlineBias)."""
    def __init__(self, factories, halflife=14.0, eta=50.0, warm=28, name="onlineens"):
        self.factories, self.halflife, self.eta, self.warm, self.name = factories, halflife, eta, warm, name

    def fit(self, tr):
        last = tr["T"].max()
        early = tr[tr["T"] <= last - pd.Timedelta(days=self.warm)]
        w = tr[tr["T"] > last - pd.Timedelta(days=self.warm)]
        self.hist = w[["T", "slot", "y", "level7"]].copy()
        for i, f in enumerate(self.factories):
            self.hist[f"p{i}"] = f().fit(early).predict(w)
        self.ms = [f().fit(tr) for f in self.factories]
        return self

    def predict(self, df):
        k = len(self.ms)
        cur = df[["T", "slot", "y", "level7"]].copy()
        for i, m in enumerate(self.ms):
            cur[f"p{i}"] = m.predict(df)
        h = pd.concat([self.hist, cur], ignore_index=True)
        h["grp"] = pd.cut(h["slot"], [0, 7, 16, 24], labels=False)
        alpha = 1 - 0.5 ** (1 / self.halflife)
        E = []
        for i in range(k):
            e = ((h[f"p{i}"] - h["y"]).abs() / h["level7"]).groupby([h["T"], h["grp"]]).mean().unstack("grp").sort_index()
            e = e.ewm(alpha=alpha, ignore_na=True).mean()
            e.index = e.index + pd.Timedelta(days=2)       # usable for day T = observed day + 2
            E.append(e)
        days = pd.DatetimeIndex(sorted(df["T"].unique()))
        E = [e.reindex(e.index.union(days)).ffill().reindex(days) for e in E]
        grp = pd.cut(df["slot"], [0, 7, 16, 24], labels=False).values
        P = cur[[f"p{i}" for i in range(k)]].values
        out = np.empty(len(df))
        Ts = pd.DatetimeIndex(df["T"])
        for j in range(len(df)):
            errs = np.array([E[i].at[Ts[j], grp[j]] for i in range(k)])
            if np.isnan(errs).any():
                w = np.ones(k) / k
            else:
                z = np.exp(-self.eta * (errs - errs.min())); w = z / z.sum()
            out[j] = P[j] @ w
        return out
