"""Sandboxed loading + validation of a dynamics candidate (Python source text).

A valid candidate defines init_state() / caution(row) / step(state, c) per the contract
in prompts/dynamics_seed.txt. compile_series_fn() turns that text into a checked
`matrix [T,6] -> control series [T]` function — the same shape as an evals integrator.
numpy + math only; dspy-free so evals/compare can import this anywhere."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

REQUIRED = ("init_state", "caution", "step")

# candidates get numpy + math + a safe builtin subset; no imports, no IO
_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum, "len": len, "range": range,
    "float": float, "int": int, "bool": bool, "round": round, "pow": pow,
    "enumerate": enumerate, "zip": zip, "sorted": sorted, "reversed": reversed,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "any": any, "all": all, "map": map, "filter": filter,
    "True": True, "False": False, "None": None,
    "Exception": Exception, "ValueError": ValueError, "isinstance": isinstance,
}


class CandidateError(Exception):
    """Compile/contract/runtime failure — the message is fed back to the reflector."""


def _exec_candidate(code: str) -> dict:
    namespace: dict = {"__builtins__": _SAFE_BUILTINS, "np": np, "math": math}
    try:
        exec(code, namespace)                                   # noqa: S102 — sandboxed on purpose
    except Exception as exc:
        raise CandidateError(f"candidate failed to exec: {type(exc).__name__}: {exc}") from None
    missing = [f for f in REQUIRED if not callable(namespace.get(f))]
    if missing:
        raise CandidateError(f"candidate must define callables {REQUIRED}; missing {missing}")
    return namespace


def compile_series_fn(code: str, validate: bool = True):
    """Candidate text -> fn(matrix [T,6]) -> control series [T]. Raises CandidateError
    with a reflector-readable message on any contract violation."""
    ns = _exec_candidate(code)
    init_state, caution_fn, step_fn = ns["init_state"], ns["caution"], ns["step"]

    def series(matrix: np.ndarray) -> np.ndarray:
        state = init_state()
        out = np.empty(len(matrix), dtype=np.float64)
        for t, row in enumerate(np.asarray(matrix, dtype=np.float64)):
            c = caution_fn(row)
            trust = step_fn(state, float(c))
            trust = float(trust)
            if not math.isfinite(trust) or not 0.0 <= trust <= 1.0:
                raise CandidateError(f"step() returned trust={trust!r} at turn {t} — must be finite in [0,1]")
            out[t] = 1.0 - trust                                # control
        return out

    if validate:
        _validate(series)
    return series


def _validate(series) -> None:
    """Contract checks on synthetic matrices. Causality needs no check — the harness
    feeds rows sequentially, so a candidate physically cannot see future turns."""
    rng = np.random.default_rng(0)
    cases = [
        np.zeros((6, 6)),                                        # all-quiet
        np.column_stack([np.ones((6, 5)), np.zeros(6)]),         # max risk, no reassurance
        np.column_stack([np.zeros((6, 5)), np.ones(6)]),         # pure reassurance
        rng.uniform(0, 1, size=(12, 6)),                         # noise
    ]
    for i, m in enumerate(cases):
        a = series(m)                                            # raises CandidateError on bound violations
        b = series(m)
        if not np.allclose(a, b):
            raise CandidateError(f"non-deterministic output on validation case {i} — randomness is forbidden")


def load_series_fn(path: Path | str):
    """Compile a saved candidate file (e.g. prompts/dynamics_optimized.txt)."""
    return compile_series_fn(Path(path).read_text(encoding="utf-8"))


def conv_integrator(path: Path | str):
    """Adapt a saved candidate to the evals integrator signature: conv dict -> series.
    Used by evals/compare to add an 'evolved' row from stored run JSONs."""
    fn = load_series_fn(path)
    return lambda conv: fn(np.asarray(conv["signals"], dtype=np.float64))
