from __future__ import annotations

import argparse
from dataclasses import replace
from types import SimpleNamespace

from data_gen.prompts.signals import LABELED_SIGNALS
from signal_layer.config import Config, load_config
from signal_layer.data import Corpus
from signal_layer.objective import TrustLayerObjective
from signal_layer.pipeline import Optimization
from signal_layer.scorer import SignalScorer, SignalsSourceLike

STAGES = ("optimize", "eval", "compare", "inspect", "dynamics-data", "optimize-dynamics")


def inspect_conversation(cfg: Config, dataset: str, conv_id: str | None) -> None:
    """Score one conversation and print the per-turn trace"""
    corpus = Corpus(cfg.data)
    conv = corpus.conversation(dataset, conv_id)
    scorer = SignalScorer.from_artifact(cfg)
    scorer.program.progress = True
    scored = scorer.score(conv)

    meta = corpus.metadata(conv["conv_id"]) or {}
    gold = SimpleNamespace(conv_id=conv["conv_id"], dataset=dataset,
                           is_adversarial=meta.get("is_adversarial", conv.get("is_adversarial", False)),
                           payload_turn=meta.get("payload_turn") if meta.get("payload_turn") is not None else -1)
    trace = TrustLayerObjective(cfg.metric).trace(scored.matrix, gold)

    print(f"conv {gold.conv_id} ({dataset}, adversarial={gold.is_adversarial}, payload_turn={gold.payload_turn})")
    print("turn  " + "  ".join(f"{name[:12]:>12}" for name in LABELED_SIGNALS) + "  control  action")
    for t, (row, turn) in enumerate(zip(scored.matrix, trace.turns)):
        print(f"{t:>4}  " + "  ".join(f"{v:>12.2f}" for v in row) + f"  {turn.control:>7.2f}  {turn.action.value}")
        if not scored.turns[t].valid:
            print(f"      (scoring failed: {scored.turns[t].error})")


def build_source(cfg: Config, args) -> tuple[SignalsSourceLike, dict]:
    """One detector per eval run, model-agnostic. Returns (source, metadata-for-the-json)."""
    if args.no_summary:                                  # turn OFF summary mode when the config defaults it on
        cfg = replace(cfg, summary_memory=False)
    if args.no_trust:                                    # isolated-turn ablation: no TRUST STATE line in context
        cfg = replace(cfg, include_trust=False)
    if args.trust_history:
        cfg = replace(cfg, trust_history=True)
    if args.summary:                                     # rolling one-line summary REPLACES the trust line
        cfg = replace(cfg, summary_memory=True, include_trust=False)
    if args.model:                                       # student-model swap (e.g. Qwen3-0.6B, Llama-3.2-1B)
        cfg = replace(cfg, student=replace(cfg.student, model=args.model))
    if args.user_only:                                    # input-filter ablation: no assistant reply in context
        cfg = replace(cfg, user_only=True)

    from signal_layer.scorer import LLMSignalsSource
    if args.artifact == "seed":                          # unoptimized seed prompt (the GEPA-off ablation)
        source = LLMSignalsSource(cfg, artifact="/nonexistent")   # from_artifact falls back to the seed
        prompt = "seed"
    else:
        source = LLMSignalsSource(cfg, artifact=args.artifact)    # None -> cfg.artifact (evolved)
        prompt = args.artifact or str(cfg.artifact)
    return source, {"system": "student", "model": cfg.student.model,
                    "prompt": prompt, "include_trust": cfg.include_trust,
                    "trust_history": cfg.trust_history, "summary_memory": cfg.summary_memory,
                    "user_only": cfg.user_only}


def main() -> None:
    ap = argparse.ArgumentParser(description="GEPA signal layer: optimize, evaluate one detector, compare runs, or inspect.")
    ap.add_argument("--stage", default="optimize", choices=STAGES)
    ap.add_argument("--config", help="path to a gepa.yaml (default: signal_layer/config/gepa.yaml)")
    ap.add_argument("--name", help="run name; output file is evals/runs/<name>.json (eval only)")
    ap.add_argument("--artifact", help="prompt artifact path, or 'seed' for the unoptimized prompt (eval only)")
    ap.add_argument("--model", help="override student model, e.g. Qwen/Qwen3-0.6B (eval only)")
    ap.add_argument("--user-only", action="store_true", help="drop the assistant reply from the rendered turn -- score the user's message alone (eval only)")
    ap.add_argument("--workers", type=int, default=1, help="conversations scored concurrently against the endpoint (eval only); e.g. 32 to saturate a vLLM server")
    ap.add_argument("--no-trust", action="store_true", help="drop the TRUST STATE line from the student's context (eval only)")
    ap.add_argument("--trust-history", action="store_true", help="show the last-4 trust values (shape) instead of the scalar alone (eval only)")
    ap.add_argument("--summary", action="store_true", help="student keeps a rolling one-line summary instead of the trust line (eval only)")
    ap.add_argument("--no-summary", action="store_true", help="force summary mode OFF when the config defaults it on (eval only)")
    ap.add_argument("--datasets", help="override evals.datasets, e.g. 'mhj=-1,wildchat=100' (eval only)")
    ap.add_argument("--dataset", default="sharegpt", help="dataset to sample from (inspect only)")
    ap.add_argument("--conv-id", help="specific conversation (inspect only; default: first in the dataset)")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()

    if args.stage == "optimize":
        Optimization(cfg).run()
    elif args.stage == "eval":
        if not args.name:
            ap.error("--stage eval requires --name (the run's filename in evals/runs/)")
        if args.datasets:
            caps = {k.strip(): int(v) for k, v in (pair.split("=") for pair in args.datasets.split(","))}
            cfg = replace(cfg, evals=replace(cfg.evals, datasets=caps))
        from signal_layer.evals.run import run_eval
        source, meta = build_source(cfg, args)
        run_eval(cfg, source, args.name, meta, workers=args.workers)
    elif args.stage == "compare":
        from signal_layer.evals.compare import compare
        compare(cfg)
    elif args.stage == "dynamics-data":
        from signal_layer.optimize.matrices import materialize
        source, meta = build_source(cfg, args)
        materialize(cfg, source, meta)
    elif args.stage == "optimize-dynamics":
        from signal_layer.optimize.evolve import run_dynamics_evolution
        run_dynamics_evolution(cfg)
    else:
        inspect_conversation(cfg, args.dataset, args.conv_id)


if __name__ == "__main__":
    main()
