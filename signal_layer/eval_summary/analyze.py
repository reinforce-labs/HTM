"""Two reports from the judge verdicts:

  1. Per judge: paired GEPA-vs-seed comparison --- mean per run, bootstrap 95% CI on the mean
     delta, and win/lose rate, for each rubric metric; overall + split by attack/benign.
  2. Across judges (when >=2 judgment files exist): inter-judge agreement per metric ---
     quadratic-weighted Cohen's kappa, Pearson correlation, exact-match rate, within-1 rate.
     This is what lets you report "two independent judges agree" rather than trusting one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from signal_layer.eval_summary.judge import METRICS


def _load(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]
    return [r for r in rows if "error" not in r]


def _boot_ci(deltas: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [rng.choice(deltas, len(deltas), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _gepa_vs_seed_block(pairs: list[tuple[dict, dict]], tag: str) -> None:
    print(f"\n  -- {tag}  (n={len(pairs)}) --")
    if not pairs:
        return
    print(f"  {'metric':<20}{'GEPA':>7}{'seed':>7}{'Δ':>8}   {'95% CI':<16} win/lose")
    for m in METRICS:
        both = [(g, s) for g, s in pairs if g.get(m) is not None and s.get(m) is not None]
        if not both:
            continue
        g = np.array([float(x[0][m]) for x in both])
        s = np.array([float(x[1][m]) for x in both])
        d = g - s
        lo, hi = _boot_ci(d)
        print(f"  {m:<20}{g.mean():>7.2f}{s.mean():>7.2f}{d.mean():>+8.2f}   "
              f"[{lo:+.2f}, {hi:+.2f}]".ljust(19) + f"{(d > 0).mean():.0%}/{(d < 0).mean():.0%}")

    def halluc_rate(idx):
        vals = [len(p[idx].get("hallucinations") or []) > 0 for p in pairs if isinstance(p[idx].get("hallucinations"), list)]
        return np.mean(vals) if vals else float("nan")
    print(f"  {'halluc. rate':<20}{halluc_rate(0):>7.0%}{halluc_rate(1):>7.0%}")


def _gepa_vs_seed(rows: list[dict], judge_name: str) -> None:
    by: dict[str, dict] = {}
    for r in rows:
        by.setdefault(r["conv_id"], {})[r["run"]] = r
    pairs = [(v["gepa"], v["seed"]) for v in by.values() if "gepa" in v and "seed" in v]
    if not pairs:
        print(f"  (no paired gepa/seed rows for judge {judge_name})")
        return
    print(f"paired conversations: {len(pairs)}   (judge scale 1-5; Δ = GEPA − seed)")
    _gepa_vs_seed_block(pairs, "all")
    _gepa_vs_seed_block([p for p in pairs if p[0]["is_adversarial"]], "attack")
    _gepa_vs_seed_block([p for p in pairs if not p[0]["is_adversarial"]], "benign")


def _weighted_kappa(a: np.ndarray, b: np.ndarray, k: int = 5) -> float:
    """Quadratic-weighted Cohen's kappa for two raters on a 1..k ordinal scale."""
    a = a.astype(int) - 1
    b = b.astype(int) - 1
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x, y] += 1
    O /= O.sum()
    hist_a, hist_b = O.sum(axis=1), O.sum(axis=0)
    E = np.outer(hist_a, hist_b)
    W = np.fromfunction(lambda i, j: (i - j) ** 2 / (k - 1) ** 2, (k, k))
    denom = (W * E).sum()
    return float(1 - (W * O).sum() / denom) if denom > 0 else float("nan")


def _agreement(rows_a: list[dict], rows_b: list[dict], name_a: str, name_b: str) -> None:
    idx_a = {(r["conv_id"], r["run"]): r for r in rows_a}
    idx_b = {(r["conv_id"], r["run"]): r for r in rows_b}
    common = sorted(set(idx_a) & set(idx_b))
    print(f"\n{'='*70}\nINTER-JUDGE AGREEMENT: {name_a}  vs  {name_b}   (n={len(common)} cells)\n{'='*70}")
    if not common:
        print("  (no overlapping judged cells)")
        return
    print(f"  {'metric':<20}{'weighted κ':>12}{'Pearson r':>12}{'exact %':>10}{'±1 %':>8}")
    for m in METRICS:
        va, vb = [], []
        for key in common:
            x, y = idx_a[key].get(m), idx_b[key].get(m)
            if x is not None and y is not None:
                va.append(x)
                vb.append(y)
        if not va:
            continue
        va_arr, vb_arr = np.array(va, dtype=float), np.array(vb, dtype=float)
        kappa = _weighted_kappa(va_arr, vb_arr)
        r = float(np.corrcoef(va_arr, vb_arr)[0, 1]) if va_arr.std() > 0 and vb_arr.std() > 0 else float("nan")
        exact = float((va_arr == vb_arr).mean())
        within1 = float((np.abs(va_arr - vb_arr) <= 1).mean())
        print(f"  {m:<20}{kappa:>12.3f}{r:>12.3f}{exact:>10.0%}{within1:>8.0%}")
    print("\n  (κ: <0 poor, 0-.2 slight, .2-.4 fair, .4-.6 moderate, .6-.8 substantial, >.8 almost perfect)")


def analyze(cfg: dict) -> None:
    judgments_dir = Path(cfg["paths"]["judgments_dir"])
    files = sorted(judgments_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no judgment files in {judgments_dir} — run --stage judge first")

    all_rows: dict[str, list[dict]] = {}
    for f in files:
        name = f.stem
        rows = _load(f)
        all_rows[name] = rows
        print(f"\n{'='*70}\nJUDGE: {name}   ({len(rows)} verdicts, {f.name})\n{'='*70}")
        _gepa_vs_seed(rows, name)

    names = list(all_rows)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            _agreement(all_rows[names[i]], all_rows[names[j]], names[i], names[j])
