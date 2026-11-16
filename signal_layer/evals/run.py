from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data_gen.prompts.signals import LABELED_SIGNALS
from signal_layer.config import Config
from signal_layer.data import Corpus
from signal_layer.evals import RUNS_DIR
from signal_layer.evals.metrics import latency_stats
from signal_layer.objective import TrustLayerObjective
from signal_layer.scorer import SignalsSourceLike


def _round(values, nd: int = 4) -> list:
    return [round(float(v), nd) for v in values]


def _unique_name(name: str) -> str:
    if not (RUNS_DIR / f"{name}.json").exists():
        return name
    n = 2
    while (RUNS_DIR / f"{name}_{n}.json").exists():
        n += 1
    return f"{name}_{n}"


def run_eval(cfg: Config, source: SignalsSourceLike, name: str, meta: dict | None = None, workers: int = 1) -> Path:
    """One detector, one JSON. workers>1 scores that many conversations concurrently against the
    signal source (turns within a conversation stay sequential — the summary memory chains them —
    but independent conversations run in parallel, so a vLLM endpoint sees many requests at once)."""
    if not cfg.evals.datasets:
        raise RuntimeError("no evals.datasets configured in gepa.yaml")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    resolved = _unique_name(name)       # resolved up front so the JSON's run.name matches its filename
    if resolved != name:                # (compare keys rows by run.name — duplicates would collide)
        print(f"[eval] {name}.json already exists -> this run will be saved as {resolved}.json")
        name = resolved

    corpus = Corpus(cfg.data)
    examples = corpus.eval_examples(cfg.evals.datasets)
    if not examples:
        raise RuntimeError(f"no eval conversations found for {dict(cfg.evals.datasets)}. Run the data_gen stages first")

    objective = TrustLayerObjective(cfg.metric)     # frozen trust layer -> control/action series
    swm = getattr(source, "signals_with_memory", None)   # thread-safe (matrix, memory); absent on sources without it

    def score_one(ex) -> tuple[dict, float]:
        start = time.perf_counter()
        if swm is not None:
            matrix, memory = swm(ex.dataset, ex.conv_id)
        else:
            matrix, memory = source.signals(ex.dataset, ex.conv_id), getattr(source, "last_memory", None)
        matrix = np.asarray(matrix, dtype=np.float64)
        seconds = time.perf_counter() - start

        trace = objective.trace(matrix, ex)     # fresh Controller per call -> thread-safe
        record = {
            "conv_id": ex.conv_id,
            "dataset": ex.dataset,
            "is_adversarial": ex.is_adversarial,
            "payload_turn": ex.payload_turn,
            "n_turns": len(matrix),
            "seconds": round(seconds, 3),
            "signals": [_round(row) for row in matrix],
            "caution": _round(t.caution for t in trace.turns),
            "trust": _round(t.trust for t in trace.turns),
            "control": _round(t.control for t in trace.turns),
            "action": [t.action.value for t in trace.turns],
        }
        if memory:                              # summary mode: per-turn case notes (qualitative evidence)
            record["memory"] = memory
        return record, seconds / max(len(matrix), 1)

    parallel = workers > 1 and swm is not None   # sources without signals_with_memory stay sequential
    if workers > 1 and swm is None:
        print("[eval] source has no signals_with_memory(); running sequentially (parallelism is for the LLM scorer)")

    results: list[tuple[dict, float]] = [None] * len(examples)
    if parallel:
        if hasattr(source, "warm"):
            source.warm(cfg.evals.datasets)      # pre-load corpus cache so threads don't race the lazy load
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(score_one, ex): i for i, ex in enumerate(examples)}
            for fut in tqdm(as_completed(futures), total=len(examples), desc=f"eval[{name}] x{workers}", unit="conv", dynamic_ncols=True, file=sys.stdout):
                results[futures[fut]] = fut.result()
    else:
        for i, ex in enumerate(tqdm(examples, desc=f"eval[{name}]", unit="conv", dynamic_ncols=True, file=sys.stdout)):
            results[i] = score_one(ex)

    conversations = [rec for rec, _ in results]
    per_turn_s = [pts for _, pts in results]

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
            "latency": latency_stats(np.asarray(per_turn_s)),
            **(meta or {}),
        },
        "conversations": conversations,
    }

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"{name}.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"[eval] {name}: {len(conversations)} convs, {payload['run']['n_turns']} turns, "
          f"{payload['run']['latency']['s_per_turn_mean']}s/turn -> {out}")
    return out
