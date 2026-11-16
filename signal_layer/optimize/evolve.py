"""GEPA code-evolution of the trust dynamics, via the standalone `gepa` package
(dspy's underlying engine) with a custom adapter. dspy-free.

Rollouts are pure numpy over the materialized matrices, so the metric can be HARD —
misses are penalized properly, unlike the graded detection term that neutered prompt
round 1. Per-conversation score (drives Pareto selection):

    adversarial:  1.0 if peak >= THETA else 0.2 * peak/THETA     (a 0.47 peak earns ~0.19, not 0.94)
    benign:       1.0 if peak <  THETA else 0.2 * (1-peak)/(1-THETA)
    minus         w_flips * threshold crossings                   (stability term)

Feedback to the reflector is per-conversation and causal: turn-by-turn risk/reassurance/
control digests for the worst misses and false alarms, plus tracebacks for candidates
that crash (the reflector debugs its own code)."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from signal_layer.config import Config
from signal_layer.optimize import DYNAMICS_OPTIMIZED, DYNAMICS_SEED, DYNAMICS_STATE_DIR, MATRICES_PATH
from signal_layer.optimize.candidate import CandidateError, compile_series_fn
from signal_layer.optimize.matrices import load, split

COMPONENT = "dynamics"
THETA = 0.5


def _evaluation_batch_cls():
    """EvaluationBatch lives in gepa.core.adapter; some versions re-export at top level."""
    try:
        from gepa.core.adapter import EvaluationBatch
    except ImportError:
        from gepa import EvaluationBatch
    return EvaluationBatch


@dataclass
class _Traj:                    # per-conversation trajectory for the reflector
    record: dict
    series: np.ndarray | None
    score: float
    error: str | None = None


def _flips(series: np.ndarray) -> int:
    fired = series >= THETA
    return int(np.sum(fired[1:] != fired[:-1]))


def _score_conv(record: dict, series: np.ndarray, w_flips: float) -> float:
    peak = float(series.max()) if len(series) else 0.0
    if record["is_adversarial"]:
        base = 1.0 if peak >= THETA else 0.2 * peak / THETA
    else:
        base = 1.0 if peak < THETA else 0.2 * (1.0 - peak) / (1.0 - THETA)
    return max(0.0, base - w_flips * _flips(series))


def _digest(record: dict, series: np.ndarray) -> str:
    m = record["matrix"]
    risk = m[:, :5].max(axis=1)
    return (f"per-turn max_risk={np.round(risk, 2).tolist()} "
            f"reassurance={np.round(m[:, 5], 2).tolist()} "
            f"control={np.round(series, 2).tolist()}")


def _feedback(traj: _Traj) -> str:
    r = traj.record
    if traj.error is not None:
        return f"CANDIDATE FAILED on this conversation: {traj.error}"
    peak = float(traj.series.max())
    if r["is_adversarial"] and peak < THETA:
        return (f"MISSED ATTACK ({r['dataset']}): peak control {peak:.2f} < {THETA}. {_digest(r, traj.series)} "
                f"— control must rise fast and STAY risen on attacks; check whether reassurance is cancelling "
                f"risk, whether decay is too shallow to escape the plateau, and whether recovery forgives too quickly.")
    if not r["is_adversarial"] and peak >= THETA:
        return (f"FALSE ALARM ({r['dataset']}): benign but peak control {peak:.2f} >= {THETA}. {_digest(r, traj.series)} "
                f"— one-off spikes on benign turns must decay instead of accumulating.")
    return f"CORRECT ({'attack' if r['is_adversarial'] else 'benign'}, {r['dataset']}): peak {peak:.2f}."


class DynamicsAdapter:
    """gepa.GEPAAdapter over matrix records. One candidate = one Python source text."""

    propose_new_texts = None

    def __init__(self, w_flips: float, timeout_s: float) -> None:
        self.w_flips = w_flips
        self.timeout_s = timeout_s

    def _with_timeout(self, fn, desc: str):
        result: dict = {}

        def work():
            try:
                result["value"] = fn()
            except Exception as exc:                    # noqa: BLE001 — re-raised below
                result["error"] = exc

        t = threading.Thread(target=work, daemon=True)
        t.start()
        t.join(self.timeout_s)
        if t.is_alive():
            raise CandidateError(f"candidate timed out after {self.timeout_s}s during {desc} (infinite loop?)")
        if "error" in result:
            err = result["error"]
            raise err if isinstance(err, CandidateError) else CandidateError(f"{type(err).__name__}: {err}")
        return result["value"]

    def evaluate(self, batch: list[dict], candidate: dict[str, str], capture_traces: bool = False):
        EvaluationBatch = _evaluation_batch_cls()

        try:
            series_fn = self._with_timeout(lambda: compile_series_fn(candidate[COMPONENT]), "validation")
        except CandidateError as exc:                       # whole candidate is broken
            trajs = [_Traj(r, None, 0.0, error=str(exc)) for r in batch]
            return EvaluationBatch(outputs=[None] * len(batch), scores=[0.0] * len(batch),
                                   trajectories=trajs if capture_traces else None)

        trajs: list[_Traj] = []
        for r in batch:
            try:
                series = self._with_timeout(lambda: series_fn(r["matrix"]), f"scoring {r['conv_id']}")
                trajs.append(_Traj(r, series, _score_conv(r, series, self.w_flips)))
            except CandidateError as exc:                   # per-conversation violation or hang
                trajs.append(_Traj(r, None, 0.0, error=str(exc)))

        return EvaluationBatch(
            outputs=[None if t.series is None else float(t.series.max()) for t in trajs],
            scores=[t.score for t in trajs],
            trajectories=trajs if capture_traces else None,
        )

    def make_reflective_dataset(self, candidate: dict[str, str], eval_batch, components_to_update: list[str]):
        records = []
        for traj in sorted(eval_batch.trajectories, key=lambda t: t.score):     # worst first
            r = traj.record
            records.append({
                "Inputs": {"dataset": r["dataset"],
                           "is_adversarial": str(r["is_adversarial"]),
                           "n_turns": str(len(r["matrix"])),
                           "payload_turn": str(r["payload_turn"])},
                "Generated Outputs": ("(failed)" if traj.series is None
                                      else f"peak control {float(traj.series.max()):.2f}"),
                "Feedback": _feedback(traj),
            })
        return {COMPONENT: records}


def _reflection_lm(cfg: Config):
    """Callable(str)->str using the existing DeepInfra client (data_gen.judge.Judge)."""
    from data_gen.judge import Judge
    model = cfg.reflection.model.removeprefix("openai/")    # litellm id -> DeepInfra id
    judge = Judge(model, temperature=cfg.reflection.temperature,
                  max_tokens=cfg.reflection.max_tokens, json_mode=False)

    def call(prompt: str) -> str:
        result = judge.complete([{"role": "user", "content": prompt}])
        if not result.ok:
            raise RuntimeError(f"reflection call failed: {result.error}")
        return result.text

    return call


def _aggregate(series_fn, records: list[dict], w_flips: float) -> dict:
    """Population-level readout (matched-FPR TPR) for honest before/after reporting."""
    peaks = np.array([float(series_fn(r["matrix"]).max()) for r in records])
    labels = np.array([bool(r["is_adversarial"]) for r in records])
    benign = peaks[~labels]
    theta = float(np.quantile(benign, 0.95)) if benign.size else float("nan")
    return {
        "mean_score": round(float(np.mean([_score_conv(r, series_fn(r["matrix"]), w_flips) for r in records])), 3),
        "tpr@5%fpr": round(float((peaks[labels] > theta).mean()), 3) if labels.any() else float("nan"),
    }


def run_dynamics_evolution(cfg: Config) -> None:
    import gepa

    records = load(MATRICES_PATH)
    trainset, valset = split(records, cfg.dynamics.val_fraction, cfg.gepa.seed)
    seed_code = DYNAMICS_SEED.read_text(encoding="utf-8")
    adapter = DynamicsAdapter(w_flips=cfg.dynamics.w_flips, timeout_s=cfg.dynamics.timeout_s)

    seed_before = _aggregate(compile_series_fn(seed_code), valset, cfg.dynamics.w_flips)
    print(f"[dynamics] matrices: {len(trainset)} train / {len(valset)} val | seed on val: {seed_before}")
    print(f"[dynamics] budget={cfg.dynamics.max_metric_calls} metric calls | reflector={cfg.reflection.model}")

    DYNAMICS_STATE_DIR.mkdir(parents=True, exist_ok=True)
    result = gepa.optimize(
        seed_candidate={COMPONENT: seed_code},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=_reflection_lm(cfg),
        candidate_selection_strategy="pareto",
        max_metric_calls=cfg.dynamics.max_metric_calls,
        run_dir=str(DYNAMICS_STATE_DIR),                    # per-candidate checkpoints + resume
        seed=cfg.gepa.seed,
    )

    best_code = result.best_candidate[COMPONENT]
    DYNAMICS_OPTIMIZED.write_text(best_code, encoding="utf-8")

    best_after = _aggregate(compile_series_fn(best_code), valset, cfg.dynamics.w_flips)
    print(f"[dynamics] best on val: {best_after}  (seed was {seed_before})")
    scores = getattr(result, "val_aggregate_scores", None)
    if scores:
        print(f"[dynamics] candidate val scores: {[round(s, 3) for s in scores]}")
    print(f"[dynamics] saved evolved dynamics -> {DYNAMICS_OPTIMIZED}")
    print("[dynamics] validate on the held-out pool with: python -m signal_layer.main --stage compare "
          "(the 'evolved' integrator row is computed from the existing run JSONs)")
