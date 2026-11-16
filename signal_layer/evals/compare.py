"""Compare all saved runs offline — no models. Each run's stored matrices are pushed
through every integrator (control | raw | max_risk | cusum), so one row per
(run x integrator) with detection, lead-time, stability, and latency columns."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from signal_layer.config import Config
from signal_layer.evals import RUNS_DIR
from signal_layer.evals.metrics import INTEGRATORS, auroc, lead_time, threshold_at_fpr, tpr_at_fpr

LEAD_FPR = 0.05     # lead time / flips are threshold-dependent; use the matched-5%-FPR threshold


def _row(convs: list[dict], integrator, fpr_points) -> dict:
    series = [integrator(c) for c in convs]
    peaks = np.array([s.max() if len(s) else 0.0 for s in series])
    labels = np.array([float(c["is_adversarial"]) for c in convs])

    theta = threshold_at_fpr(peaks, labels, LEAD_FPR)
    leads = [lead_time(s, c["payload_turn"], theta) for s, c in zip(series, convs) if c["is_adversarial"]]
    leads = [l for l in leads if l is not None]

    row = {"auroc": round(auroc(peaks, labels), 3)}
    for f in fpr_points:
        row[f"tpr@{f:g}fpr"] = round(tpr_at_fpr(peaks, labels, f), 3)
    row["mean_lead"] = round(float(np.mean(leads)), 2) if leads else None
    return row


def _integrators() -> dict:
    return dict(INTEGRATORS)     # control + max_risk only


def compare(cfg: Config) -> dict:
    runs = sorted(RUNS_DIR.glob("*.json"))
    runs = [p for p in runs if p.name != "compare.json"]
    if not runs:
        raise RuntimeError(f"no runs in {RUNS_DIR} — run --stage eval first")

    fpr_points = cfg.evals.fpr_points
    integrators = _integrators()
    table: dict[str, dict] = {}
    for path in runs:
        data = json.loads(path.read_text(encoding="utf-8"))
        convs, run = data["conversations"], data["run"]
        for iname, integrator in integrators.items():
            row = _row(convs, integrator, fpr_points)
            row["s/turn"] = run["latency"]["s_per_turn_mean"]
            row["n"] = run["n_conversations"]
            table[f"{run['name']}/{iname}"] = row

    # aligned table: rows sorted by auroc so the leaderboard reads top-down
    cols = ["auroc"] + [f"tpr@{f:g}fpr" for f in fpr_points] + ["mean_lead", "s/turn", "n"]
    width = max(len(k) for k in table) + 2
    print(f"{'run/integrator':<{width}}" + "".join(f"{c:>12}" for c in cols))
    for key, row in sorted(table.items(), key=lambda kv: -(kv[1]["auroc"] or 0)):
        print(f"{key:<{width}}" + "".join(f"{str(row.get(c, '-')):>12}" for c in cols))

    out = RUNS_DIR / "compare.json"
    out.write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(f"[compare] wrote {out}")
    return table
