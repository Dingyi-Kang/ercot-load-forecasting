"""Build reports/benchmark_validation.md from experiments/registry.jsonl.
All runs are listed (including failures and superseded reruns), in execution order."""
import json
from pathlib import Path

GAP = ("2023-12-30", "2024-01-19")


def main():
    rows = [json.loads(l) for l in Path("experiments/registry.jsonl").read_text().splitlines() if l.strip()]
    val = [r for r in rows if r["phase"] == "validation"]
    lines = ["# Validation benchmark (2024-01-01 .. 2025-12-31, 8 quarterly rolling-origin folds)", "",
             "Generated from experiments/registry.jsonl by src/elf/report.py.",
             f"Validation runs so far: {len(val)} (budget 60).", "",
             "| # | experiment | status | MAE | RMSE | MAPE % | bias | peak MAE | high-load MAE |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(val, 1):
        m = r["metrics"] or {}
        f = lambda k, d=0: (f"{m[k]:.{d}f}" if m.get(k) is not None else "–")
        lines.append(f"| {i} | {r['experiment']} | {'ok' if r['status']=='ok' else 'error'} | {f('mae')} | {f('rmse')} | {f('mape',2)} | "
                     f"{f('bias')} | {f('peak_mag_mae')} | {f('high_load_mae')} |")
    lines += ["", "Hypotheses for each experiment are stored in the registry and in src/elf/experiments.py."]
    ho = [r for r in rows if r["phase"] == "holdout"]
    if ho:
        lines += ["", "# Final holdout (2026-01-01 .. 2026-09-30, opened once after configs/final.yaml was locked)", "",
                  "| experiment | MAE | RMSE | MAPE % | bias | peak MAE | high-load MAE |", "|---|---|---|---|---|---|---|"]
        for r in ho:
            m = r["metrics"] or {}
            f = lambda k, d=0: (f"{m[k]:.{d}f}" if m.get(k) is not None else "–")
            lines.append(f"| {r['experiment']} | {f('mae')} | {f('rmse')} | {f('mape',2)} | {f('bias')} | {f('peak_mag_mae')} | {f('high_load_mae')} |")
    Path("reports/benchmark_validation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-len(val) - 2:]))


if __name__ == "__main__":
    main()
