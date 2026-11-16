"""Model-agnostic evaluation for the GEPA signal layer.

Two-step design (mirrors eval/benchmark): RUN each detector one at a time, persisting
everything per conversation — the [T,6] signal matrix, the trust-layer series (caution/
trust/control/action), per-turn validity, and latency — to evals/runs/<name>.json; then
COMPARE across all saved runs offline (metrics + alternative integrators like raw-signal
and CUSUM are computed from the stored matrices, no model re-runs needed).

    run.py       score one SignalsSourceLike over the configured datasets -> runs/<name>.json
    metrics.py   pure functions: auroc, tpr@fpr, lead time, flips, integrators (numpy only)
    compare.py   read runs/*.json -> one table across (run x integrator), writes compare.json

CLI:  python -m signal_layer.main --stage eval --name gepa_trust
      python -m signal_layer.main --stage eval --name seed --artifact seed
      python -m signal_layer.main --stage eval --name qwen0.6b --model Qwen/Qwen3-0.6B
      python -m signal_layer.main --stage compare
"""

from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent / "runs"   # here (not in run.py) so compare.py stays dspy-free
