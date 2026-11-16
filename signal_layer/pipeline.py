from __future__ import annotations

import sys
from pathlib import Path

import dspy
from tqdm import tqdm

from signal_layer.config import Config
from signal_layer.data import Corpus
from signal_layer.objective import TrustLayerObjective
from signal_layer.scorer import SignalScorer, build_lm


class _LoggingLM(dspy.LM):
    """Reflection LM that prints on every call — each one is a paid Claude reflection
    (prompt rewrite). Lets you watch the Sonnet spend accrue in real time."""

    _calls = 0

    def __call__(self, *args, **kwargs):
        type(self)._calls += 1
        tqdm.write(f"[reflect] Claude reflection call #{type(self)._calls}", file=sys.stdout)
        return super().__call__(*args, **kwargs)


def _summarize_split(name: str, examples: list) -> None:
    adversarial = sum(1 for ex in examples if ex.is_adversarial)
    labeled = sum(1 for ex in examples if ex.teacher)
    print(f"[data] {name:<8} {len(examples):>5} convs | adversarial={adversarial} benign={len(examples) - adversarial} "
          f"| teacher-labeled={labeled}")


class Optimization:
    """One GEPA run: evolve the scoring instructions on the feedback set, select
    candidates on the Pareto set, save the winner to cfg.artifact.
    (Held-out evaluation lives in signal_layer/evals — run one detector at a time with
    `--stage eval --name <run>`, then `--stage compare` across the saved runs.)"""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.corpus = Corpus(cfg.data)
        self.scorer = SignalScorer(cfg)                     # seed instructions, not the artifact
        self.best_dir = cfg.artifact.parent / "best_prompt"
        self.scorer.program.checkpoint_dir = self.best_dir  # captures each candidate on first evaluation

        # ground-truth rollout counter: fires from inside the metric on every REAL scored
        # conversation, independent of dspy's own internal (estimated) progress accounting
        self._bar = tqdm(desc="rollouts", unit="conv", dynamic_ncols=True, file=sys.stdout)
        self.objective = TrustLayerObjective(cfg.metric, on_score=self._record_rollout)

    def _record_rollout(self, gold: dspy.Example, score: float) -> None:
        self._bar.update(1)
        self._bar.set_postfix_str(f"last={score:.2f} {gold.dataset}")

    def run(self) -> Path:
        feedback, pareto = self.corpus.feedback(), self.corpus.pareto()
        _summarize_split("feedback", feedback)      # GEPA trainset: reflector reads these failures
        _summarize_split("pareto", pareto)          # GEPA valset: candidates selected on these
        _summarize_split("eval", self.corpus.eval_examples(self.cfg.evals.datasets))  # held out for --stage eval
        print(f"[gepa] include_trust={self.cfg.include_trust} student={self.cfg.student.model} "
              f"reflector={self.cfg.reflection.model} auto={self.cfg.gepa.auto}")

        gepa_state_dir = self.best_dir / "gepa_state"
        gepa_state_dir.mkdir(parents=True, exist_ok=True)
        print(f"[gepa] checkpointing every candidate to {gepa_state_dir} "
              f"(resumable: rerun with this same log_dir to continue after a crash)")

        gepa = dspy.GEPA(
            metric=self.objective,
            reflection_lm=build_lm(self.cfg.reflection, cls=_LoggingLM),   # prints on every Claude call
            auto=self.cfg.gepa.auto,
            candidate_selection_strategy="pareto",          # greedy converges to a local optimum after one iteration
            reflection_minibatch_size=self.cfg.gepa.reflection_minibatch_size,
            use_merge=self.cfg.gepa.use_merge,
            num_threads=self.cfg.gepa.threads,
            track_stats=True,
            log_dir=str(gepa_state_dir),                    # native per-candidate checkpoint + resume
            seed=self.cfg.gepa.seed,
        )
        try:
            self.scorer.program = gepa.compile(self.scorer.program, trainset=feedback, valset=pareto)
        finally:
            self._bar.close()

        path = self.scorer.save()
        print(f"[gepa] saved evolved program -> {path}")

        results = getattr(self.scorer.program, "detailed_results", None)
        if results is not None:
            print(f"[gepa] best pareto aggregate: {results.best_idx=} "
                  f"val_scores={[round(s, 3) for s in results.val_aggregate_scores]}")
        return path
