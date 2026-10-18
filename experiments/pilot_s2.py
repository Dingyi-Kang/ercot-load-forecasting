"""Study S2 pilot (PROTOCOL A4/A5): choose Chronos-2 fine-tuning settings on the
training period only. Fine-tune on data <= 2023-06-30; score target days 2023-07-01..2023-12-28.
Usage: python -m experiments.pilot_s2 zero
       python -m experiments.pilot_s2 MODE STEPS LR [--bs N] [--seed S] [--ztemp] [--r R]
Predictions are saved to experiments/pilot_preds/<name>.parquet (gitignored) for seed-averaging checks."""
import argparse, time, json
from pathlib import Path
import numpy as np, pandas as pd
from src.elf.tsfm import Chronos2Zero, Chronos2FT

ap = argparse.ArgumentParser()
ap.add_argument("mode"); ap.add_argument("steps", nargs="?", type=int); ap.add_argument("lr", nargs="?", type=float)
ap.add_argument("--bs", type=int, default=32); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ztemp", action="store_true"); ap.add_argument("--r", type=int, default=None)
a = ap.parse_args()

load = pd.read_parquet("data/processed/load_hourly.parquet")
load["op_date"] = pd.to_datetime(load["op_date"])
te = load[(load.op_date >= "2023-07-01") & (load.op_date <= "2023-12-28")].rename(columns={"op_date": "T"})
tr = pd.DataFrame({"T": [pd.Timestamp("2023-06-29")]})          # last training target day = S-2

t0 = time.time()
if a.mode == "zero":
    m = Chronos2Zero(covariates=True, blend=True)
else:
    m = Chronos2FT(mode=a.mode, steps=a.steps, lr=a.lr, blend=True, batch_size=a.bs, seed=a.seed,
                   zone_temp=a.ztemp, lora_r=a.r)
    m.fit(tr)
fit_s = time.time() - t0
p = m.predict(te)
out = Path("experiments/pilot_preds"); out.mkdir(exist_ok=True)
pd.DataFrame({"interval_start_utc": te["interval_start_utc"].values, "T": te["T"].values,
              "y": te["ERCOT"].values, "pred": p}).to_parquet(out / f"{m.name}.parquet", index=False)
err = p - te["ERCOT"].values
res = dict(config=m.name, fit_s=round(fit_s), n=len(te), mae=round(float(np.abs(err).mean()), 1),
           bias=round(float(err.mean()), 1), mae_by_month={str(k): round(float(v), 1) for k, v in
           pd.Series(np.abs(err)).groupby(te["T"].dt.month.values).mean().items()})
print(json.dumps(res))
with open("experiments/exploratory_notes.txt", "a") as f:
    f.write(f"S2 pilot {time.strftime('%Y-%m-%d %H:%M')}: {json.dumps(res)}\n")
