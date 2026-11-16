"""Score a single-turn guard (Granite / Llama Guard) over the SAME held-out eval set as
the model runs, so it lands as a baseline row in the same compare.py leaderboard.

A guard has no controller: its per-turn P(unsafe) IS the score series. We store that series
as `control` (what series_control reads) and broadcast it across a [T,6] `signals` matrix so
series_max_risk returns the same peak — a guard row is identical under both integrators.
The eval set comes from the same Corpus.eval_examples(cfg.evals.datasets) call run.py makes,
seeded identically, so the conversations match final_gepa_summary.json turn for turn.

    python -m signal_layer.evals.run_guard --guard granite --variant current
    python -m signal_layer.evals.run_guard --guard granite --variant history --name granite_history
    python -m signal_layer.main --stage compare      # guard row now joins the table

Granite (and Llama Guard) can also be served on vLLM instead of loaded in-process. The prompt
is built client-side with the guardian_config template and sent to /completions as token ids,
so vLLM never applies a chat template:

    vllm serve ibm-granite/granite-guardian-3.2-5b --port 8001
    python -m signal_layer.evals.run_guard --guard granite --variant prompt \\
        --vllm-base http://localhost:8001/v1 --name granite_user_only
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data_gen.prompts.signals import LABELED_SIGNALS
from data_gen.schema import Turn
from eval.baseline.device import get_device
from eval.baseline.guard_scores import GUARD_VARIANTS, GUARDS
from signal_layer.config import Config, load_config
from signal_layer.data import Corpus
from signal_layer.evals import RUNS_DIR
from signal_layer.evals.metrics import auroc, latency_stats, lead_time, threshold_at_fpr, tpr_at_fpr


def _turns(ex) -> list[Turn]:
    return [Turn(user=t.get("user"), assistant=t.get("assistant")) for t in ex.conv["turns"]]


def build_guard(guard_name: str, device: str, vllm_base: str | None, served_model: str | None):
    """The served (vLLM) or in-process (transformers) guard for guard_name."""
    if vllm_base:
        if guard_name == "llama_guard":
            from eval.baseline.classifiers.vllm_guard import MODEL_ID, VLLMLlamaGuard
            print(f"[guard] using vLLM endpoint {vllm_base} (model {served_model or MODEL_ID})")
            return VLLMLlamaGuard(api_base=vllm_base, model=served_model or MODEL_ID)
        if guard_name == "granite":
            from eval.baseline.classifiers.vllm_guard import GRANITE_MODEL_ID, VLLMGraniteGuardian
            print(f"[guard] using vLLM endpoint {vllm_base} (model {served_model or GRANITE_MODEL_ID})")
            return VLLMGraniteGuardian(api_base=vllm_base, model=served_model or GRANITE_MODEL_ID)
        raise SystemExit(f"--vllm-base is wired for llama_guard and granite only, not {guard_name!r}")

    device = get_device(device)
    print(f"[guard] loading {guard_name} in-process on {device}")
    return GUARDS[guard_name](device=device)


def run_guard(cfg: Config, guard_name: str, variant: str, name: str, batch_size: int, device: str,
              vllm_base: str | None = None, served_model: str | None = None) -> Path:
    """One guard, one variant, one JSON — same eval set and schema as signal_layer/evals/run.py."""
    if guard_name not in GUARDS:
        raise SystemExit(f"unknown guard {guard_name!r}; choose from {sorted(GUARDS)}")
    if variant not in GUARD_VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}; choose from {sorted(GUARD_VARIANTS)}")

    corpus = Corpus(cfg.data)
    examples = corpus.eval_examples(cfg.evals.datasets)
    if not examples:
        raise RuntimeError("no eval conversations found — run the data_gen stages first")

    build = GUARD_VARIANTS[variant]
    turns_by = {ex.conv_id: _turns(ex) for ex in examples}

    jobs, meta = [], []                                  # one (conv, turn) cell per guard forward
    for ex in examples:
        for t in range(len(turns_by[ex.conv_id])):
            jobs.append(build(turns_by[ex.conv_id], t))
            meta.append((ex.conv_id, t))

    print(f"[guard] {guard_name}/{variant}: {len(jobs)} cells across {len(examples)} convs")
    guard = build_guard(guard_name, device, vllm_base, served_model)

    start = time.perf_counter()
    scored = guard.score_batch(jobs, batch_size=batch_size, desc=f"{guard_name}/{variant}")
    per_turn_s = (time.perf_counter() - start) / max(len(jobs), 1)

    cell = {(cid, t): float(s) for (cid, t), s in zip(meta, scored)}

    conversations = []
    for ex in examples:
        n = len(turns_by[ex.conv_id])
        series = [round(cell[(ex.conv_id, t)], 4) for t in range(n)]
        conversations.append({
            "conv_id": ex.conv_id,
            "dataset": ex.dataset,
            "is_adversarial": ex.is_adversarial,
            "payload_turn": ex.payload_turn,
            "n_turns": n,
            "seconds": round(per_turn_s * n, 3),
            "signals": [[s] * len(LABELED_SIGNALS) for s in series],   # broadcast -> max_risk == control
            "control": series,
        })

    counts: dict[str, int] = {}
    for c in conversations:
        counts[c["dataset"]] = counts.get(c["dataset"], 0) + 1

    payload = {
        "run": {
            "name": name,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "signal_names": list(LABELED_SIGNALS),
            "datasets": counts,
            "n_conversations": len(conversations),
            "n_turns": int(sum(c["n_turns"] for c in conversations)),
            "latency": latency_stats(np.full(len(jobs), per_turn_s)),
            "system": "guard",
            "guard": guard_name,
            "variant": variant,
        },
        "conversations": conversations,
    }

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"{name}.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    _print_metrics(conversations, cfg.evals.fpr_points, name)
    print(f"[guard] {name}: {len(conversations)} convs, {payload['run']['n_turns']} turns -> {out}")
    return out


def _print_metrics(conversations: list[dict], fpr_points, name: str) -> None:
    """Same metrics compare.py reports, printed up front so no second command is needed."""
    peaks = np.array([max(c["control"]) if c["control"] else 0.0 for c in conversations])
    labels = np.array([float(c["is_adversarial"]) for c in conversations])

    theta = threshold_at_fpr(peaks, labels, 0.05)
    leads = [lead_time(np.asarray(c["control"]), c["payload_turn"], theta)
             for c in conversations if c["is_adversarial"]]
    leads = [l for l in leads if l is not None]

    print(f"\n[{name}]  AUROC={auroc(peaks, labels):.3f}", end="")
    for f in fpr_points:
        print(f"  TPR@{f:g}={tpr_at_fpr(peaks, labels, f):.3f}", end="")
    print(f"  mean_lead={np.mean(leads):.2f}" if leads else "  mean_lead=n/a")


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a single-turn guard over the signal_layer held-out eval set.")
    ap.add_argument("--guard", required=True, choices=sorted(GUARDS), help="granite | llama_guard")
    ap.add_argument("--variant", default="current", choices=sorted(GUARD_VARIANTS),
                    help="current: this turn's user+assistant (fair single-turn); history: full transcript through t; prompt: user only")
    ap.add_argument("--name", help="run name -> evals/runs/<name>.json (default: <guard>_<variant>)")
    ap.add_argument("--batch-size", type=int, default=16, help="in-process batch size (ignored for --vllm-base)")
    ap.add_argument("--device", default="auto", help="in-process device (ignored for --vllm-base)")
    ap.add_argument("--vllm-base", help="score against a vLLM OpenAI endpoint, e.g. http://localhost:8001/v1 (llama_guard or granite)")
    ap.add_argument("--served-model", help="model name the vLLM server reports (default: the guard's own MODEL_ID)")
    ap.add_argument("--config", help="path to a gepa.yaml (default: signal_layer/config/gepa.yaml)")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    name = args.name or f"{args.guard}_{args.variant}"
    run_guard(cfg, args.guard, args.variant, name, args.batch_size, args.device, args.vllm_base, args.served_model)


if __name__ == "__main__":
    main()
