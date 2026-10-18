"""Zero-shot time-series foundation model (Chronos-2) under the A0 protocol.

For target day T (issue 18:00 on D=T-1): context = hourly ERCOT load up to the end of operating
day T-2 (local), last `context_hours` hours. The model predicts every hour from there to the end of
day T (~48 h); only day T is scored. Optional covariates (known future): GFS weighted temperature
(per-hour vintage rule relative to T's issue; for context hours the day-1 vintage, all of which are
older than the issue), plus calendar (local hour, day of week, holiday flag).
No training: `fit` is a no-op (zero-shot), so there is no fold-level leakage path.
"""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from . import weather as W
from .weather_download import CITIES
from .features import holiday_table

TZ = "America/Chicago"


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class Chronos2Zero:
    def __init__(self, covariates=True, context_hours=2048, model_id="amazon/chronos-2", name=None, blend=False):
        self.cov, self.ctx, self.model_id, self.blend = covariates, context_hours, model_id, blend
        self.name = name or f"chronos2_{'cov' if covariates else 'uni'}_{context_hours}{'_blend' if blend else ''}"
        self._pipe = None

    def fit(self, tr):
        return self

    # ------------------------------------------------------------------ data helpers
    def _prep(self):
        if hasattr(self, "_load"):
            return
        load = pd.read_parquet("data/processed/load_hourly.parquet")
        self._load = load.set_index("interval_start_utc")["ERCOT"].sort_index()
        self._opday = load.set_index("interval_start_utc")["op_date"].astype(str)
        if self.cov:
            wts = W.city_weights(load)
            tab = W.fcst_table("gfs_global")
            Wv = np.array([wts[c] for c in CITIES])
            d1 = np.column_stack([tab["temperature_2m_previous_day1"].xs(c, level="city").reindex(self._load.index) for c in CITIES])
            d2 = np.column_stack([tab["temperature_2m_previous_day2"].xs(c, level="city").reindex(self._load.index) for c in CITIES])
            wavg = lambda M: (np.nan_to_num(M) * Wv).sum(1) / (~np.isnan(M) * Wv).sum(1)
            self._t1 = pd.Series(wavg(d1), index=self._load.index)
            self._d1c = pd.DataFrame(d1, index=self._load.index, columns=list(CITIES))
            self._t2 = pd.Series(wavg(d2), index=self._load.index)
            if self.blend:   # ECMWF IFS 0.25 (2024-02+); averaged with GFS where available
                et = W.fcst_table("ecmwf_ifs025")
                e1 = np.column_stack([et["temperature_2m_previous_day1"].xs(c, level="city").reindex(self._load.index) for c in CITIES])
                e2 = np.column_stack([et["temperature_2m_previous_day2"].xs(c, level="city").reindex(self._load.index) for c in CITIES])
                self._e1 = pd.Series(wavg(e1), index=self._load.index)
                self._e2 = pd.Series(wavg(e2), index=self._load.index)
            hol = holiday_table("2016-01-01", "2027-01-01")
            hol.index = pd.to_datetime(hol.index)
            self._hol = hol["holiday"] != "none"

    def _calendar(self, idx):
        loc = idx.tz_convert(TZ)
        days = pd.to_datetime(loc.tz_localize(None).normalize())
        return pd.DataFrame({"hour": loc.hour.values, "dow": loc.dayofweek.values,
                             "hol": self._hol.reindex(days).fillna(False).astype(int).values}, index=idx)

    def _item(self, T):
        T = pd.Timestamp(T)
        end_ctx = (T - pd.Timedelta(days=1)).tz_localize(TZ).tz_convert("UTC")      # end of T-2 = start of T-1
        end_tgt = (T + pd.Timedelta(days=1)).tz_localize(TZ).tz_convert("UTC")      # end of T
        issue = (T - pd.Timedelta(days=1) + pd.Timedelta(hours=18)).tz_localize(TZ).tz_convert("UTC")
        ctx_idx = pd.date_range(end_ctx - pd.Timedelta(hours=self.ctx), end_ctx, freq="h", inclusive="left")
        fut_idx = pd.date_range(end_ctx, end_tgt, freq="h", inclusive="left")
        past = pd.DataFrame({"item_id": str(T.date()), "timestamp": ctx_idx.tz_localize(None),
                             "target": self._load.reindex(ctx_idx).values})
        fut = pd.DataFrame({"item_id": str(T.date()), "timestamp": fut_idx.tz_localize(None)})
        if self.cov:
            # context temperature: day-1 vintage (all older than the issue); future: freshest allowed
            past["temp"] = self._t1.reindex(ctx_idx).values
            ends = fut_idx + pd.Timedelta(hours=1)
            ok1 = (ends - pd.Timedelta(hours=24)).floor("6h") + pd.Timedelta(hours=W.PUB_DELAY_H) <= issue
            t1, t2 = self._t1.reindex(fut_idx).values, self._t2.reindex(fut_idx).values
            fut["temp"] = np.where(ok1 & ~np.isnan(t1), t1, t2)
            if self.blend:
                de = pd.Timedelta(hours=W.PUB_DELAY_BY_MODEL["ecmwf_ifs025"])
                ok1e = (ends - pd.Timedelta(hours=24)).floor("6h") + de <= issue
                e1, e2 = self._e1.reindex(fut_idx).values, self._e2.reindex(fut_idx).values
                ef = np.where(ok1e & ~np.isnan(e1), e1, e2)
                fut["temp"] = np.where(np.isnan(ef), fut["temp"].values, (ef + fut["temp"].values) / 2)
                ec = self._e1.reindex(ctx_idx).values
                past["temp"] = np.where(np.isnan(ec), past["temp"].values, (ec + past["temp"].values) / 2)
            for k, v in self._calendar(ctx_idx).items():
                past[k] = v.values
            for k, v in self._calendar(fut_idx).items():
                fut[k] = v.values
            past["temp"] = past["temp"].interpolate(limit_direction="both")
        return past, fut, fut_idx

    # ------------------------------------------------------------------ predict
    def predict(self, df):
        self._prep()
        if self._pipe is None:
            import torch
            from chronos import Chronos2Pipeline
            torch.set_num_threads(16)
            self._pipe = Chronos2Pipeline.from_pretrained(self.model_id, device_map=_device())
        days = sorted(pd.to_datetime(df["T"]).unique())
        out = pd.Series(np.nan, index=pd.DatetimeIndex(df["interval_start_utc"]))
        preds = {}
        # group days by prediction length (47/48/49 on DST) because predict_df needs a fixed length
        items = {}
        for T in days:
            p, f, idx = self._item(T)
            items.setdefault(len(idx), []).append((p, f, idx, T))
        for H, lst in items.items():
            for i in range(0, len(lst), 64):
                chunk = lst[i:i + 64]
                past = pd.concat([c[0] for c in chunk], ignore_index=True)
                fut = pd.concat([c[1] for c in chunk], ignore_index=True)
                if not self.cov:
                    fut = None
                r = self._pipe.predict_df(past, future_df=fut, prediction_length=H, quantile_levels=[0.5], context_length=self.ctx,
                                          id_column="item_id", timestamp_column="timestamp", target="target")
                col = "predictions" if "predictions" in r.columns else "0.5"
                for (_, _, idx, T) in chunk:
                    v = r[r["item_id"] == str(pd.Timestamp(T).date())][col].values
                    start_T = pd.Timestamp(T).tz_localize(TZ).tz_convert("UTC")
                    sr = pd.Series(v, index=idx)
                    preds[T] = sr[sr.index >= start_T]   # score only operating day T
        allp = pd.concat(preds.values())
        return allp.reindex(pd.DatetimeIndex(df["interval_start_utc"])).values


class Chronos2FT(Chronos2Zero):
    """Chronos-2 fine-tuned (LoRA) at each fold start on data available then (transfer learning).
    Training series: ERCOT total + 8 weather zones, hourly, from `ft_start` to the end of the last
    training day (<= S-2), with known-future covariates (GFS day-1 temperature, hour, dow, holiday).
    The fold-level fit happens once; predictions then use the fine-tuned weights zero-shot style."""
    ZONES = ["COAST", "EAST", "FWEST", "NORTH", "NCENT", "SOUTH", "SCENT", "WEST"]

    def __init__(self, steps=300, lr=1e-5, batch_size=32, ft_context=2048, ft_start="2021-04-01", mode="lora", seed=0,
                 zone_temp=False, lora_r=None, cache=False, eval_fix=True, **kw):
        super().__init__(covariates=True, **kw)
        self.steps, self.lr, self.bs, self.ft_ctx, self.ft_start, self.mode, self.seed = steps, lr, batch_size, ft_context, ft_start, mode, seed
        self.zone_temp, self.lora_r, self.cache, self.eval_fix = zone_temp, lora_r, cache, eval_fix
        self.name = (f"chronos2_ft_{mode}_{steps}_lr{lr:g}{'_blend' if self.blend else ''}"
                     f"{'_ztemp' if zone_temp else ''}{f'_r{lora_r}' if lora_r else ''}{f'_s{seed}' if seed else ''}")

    def fit(self, tr):
        import torch
        from chronos import Chronos2Pipeline
        from chronos.chronos2.preprocess import from_data_frame
        self._prep()
        torch.set_num_threads(16)
        last_day = pd.Timestamp(tr["T"].max())
        out_dir = f"{tempfile.gettempdir()}/chronos_ft/{self.name}_{last_day.date()}"
        ckpt = Path(out_dir) / "finetuned-ckpt"
        if self.cache and (ckpt / "config.json").exists():
            # same config, seed and training cutoff -> reuse the fine-tuned weights (compute saving only)
            self._pipe = Chronos2Pipeline.from_pretrained(str(ckpt), device_map=_device())
            return self
        torch.manual_seed(self.seed)
        base = Chronos2Pipeline.from_pretrained(self.model_id, device_map=_device())
        end = (last_day + pd.Timedelta(days=1)).tz_localize(TZ).tz_convert("UTC")
        start = pd.Timestamp(self.ft_start).tz_localize(TZ).tz_convert("UTC")
        idx = pd.date_range(start, end, freq="h", inclusive="left")
        load = pd.read_parquet("data/processed/load_hourly.parquet").set_index("interval_start_utc")
        cal = self._calendar(idx)
        temp = self._t1.reindex(idx).interpolate(limit_direction="both").values
        parts = []
        for col in ["ERCOT"] + self.ZONES:
            t = temp
            if self.zone_temp and col != "ERCOT":   # zone series get their own cities' temperature
                cs = [c for c, v in CITIES.items() if v[2] == col]
                t = self._d1c[cs].mean(axis=1).reindex(idx).interpolate(limit_direction="both").values
            parts.append(pd.DataFrame({"item_id": col, "timestamp": idx.tz_localize(None),
                                       "target": load[col].reindex(idx).values, "temp": t,
                                       "hour": cal["hour"].values, "dow": cal["dow"].values, "hol": cal["hol"].values}))
        df = pd.concat(parts, ignore_index=True)
        inputs = from_data_frame(df, target_columns=["target"], prediction_length=48,
                                 known_covariates_names=["temp", "hour", "dow", "hol"])
        extra = {}
        if self.mode == "lora" and self.lora_r:
            extra["lora_config"] = dict(r=self.lora_r, lora_alpha=2 * self.lora_r, target_modules=[
                "self_attention.q", "self_attention.v", "self_attention.k", "self_attention.o",
                "output_patch_embedding.output_layer"])
        self._pipe = base.fit(inputs, prediction_length=48, finetune_mode=self.mode, learning_rate=self.lr, **extra,
                              num_steps=self.steps, batch_size=self.bs, context_length=self.ft_ctx,
                              output_dir=out_dir, remove_printer_callback=True,
                              save_strategy="no", seed=self.seed)
        if self.eval_fix:
            # Chronos2Pipeline.fit returns the model still in train mode (dropout active at inference);
            # found 2026-10-14 (PROTOCOL A5). T5/T6/E11/E12 ran without this fix.
            self._pipe.model.eval()
        return self


class Chronos2FTSeeds:
    """Average of several Chronos2FT fits that differ only in the random seed (variance reduction;
    PROTOCOL A5). Each member is fine-tuned on exactly the same data/cutoff."""
    def __init__(self, seeds=(0, 1, 2), **kw):
        self.ms = [Chronos2FT(seed=s, cache=True, **kw) for s in seeds]
        self.name = self.ms[0].name + f"_seeds{len(seeds)}"

    def fit(self, tr):
        for m in self.ms:
            m.fit(tr)
        return self

    def predict(self, df):
        return np.mean([m.predict(df) for m in self.ms], axis=0)
