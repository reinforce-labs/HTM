from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import yaml

from data_gen.io import REPO_ROOT

CONFIG_PATH = Path(__file__).resolve().parent / "gepa.yaml"


@dataclass(frozen=True)
class LMCfg:
    model: str                                          # litellm id (openai/...) for API backends; HF repo id when local
    temperature: float
    max_tokens: int
    api_base: str = ""                                  # API backends only
    api_key_env: str = ""                               # env var holding the key ("local" fallback for localhost)
    local: bool = False                                 # in-process transformers on this machine's GPU (no server)


@dataclass(frozen=True)
class GepaCfg:
    auto: str = "light"                                 # light | medium | heavy
    reflection_minibatch_size: int = 3
    use_merge: bool = False
    threads: int = 8
    seed: int = 0


@dataclass(frozen=True)
class MetricCfg:
    detection_weight: float = 0.6                       # trajectory gold via the frozen trust layer
    agreement_weight: float = 0.4                       # per-turn teacher closeness
    threshold: float = 0.5                              # control U separating adversarial from benign
    wrong_side_credit: float = 0.2                      # credit multiplier on the WRONG side of the threshold.


@dataclass(frozen=True)
class DataCfg:
    adversarial: tuple[str, ...]                        # optimization pools (role == train)
    benign: tuple[str, ...]
    test_adversarial: tuple[str, ...]                   # held-out evaluation
    test_benign: tuple[str, ...]
    feedback_size: int = 64
    pareto_size: int = 200
    seed: int = 0
    optimization_caps: dict[str, int] = None 
    synth_split: dict[str, int] = None


@dataclass(frozen=True)
class EvalsCfg:
    datasets: dict[str, int]
    fpr_points: tuple[float, ...] = (0.01, 0.05)


@dataclass(frozen=True)
class DynamicsCfg:
    max_metric_calls: int = 600
    val_fraction: float = 0.3
    timeout_s: float = 20.0
    w_flips: float = 0.02


@dataclass(frozen=True)
class Config:
    include_trust: bool
    student: LMCfg
    reflection: LMCfg
    gepa: GepaCfg
    metric: MetricCfg
    data: DataCfg
    evals: EvalsCfg
    dynamics: DynamicsCfg
    artifact: Path
    trust_history: bool = False
    summary_memory: bool = False
    user_only: bool = False


def _lm(raw: dict) -> LMCfg:
    cfg = LMCfg(
        model=raw["model"],
        temperature=float(raw["temperature"]),
        max_tokens=int(raw["max_tokens"]),
        api_base=raw.get("api_base", ""),
        api_key_env=raw.get("api_key_env", ""),
        local=bool(raw.get("local", False)),
    )
    if not cfg.local and not cfg.api_base:
        raise ValueError(f"{cfg.model}: api_base is required unless local: true")
    return cfg


def _data(raw: dict) -> DataCfg:
    lists = {k: tuple(raw[k]) for k in ("adversarial", "benign", "test_adversarial", "test_benign")}
    sizes = {k: int(raw[k]) for k in ("feedback_size", "pareto_size", "seed")}
    caps = {k: int(v) for k, v in (raw.get("optimization_caps") or {}).items()}
    synth = {k: int(v) for k, v in (raw.get("synth_split") or {}).items()}
    return DataCfg(**lists, **sizes, optimization_caps=caps, synth_split=synth)


def load_config(path: Path | str = CONFIG_PATH) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    include_trust = raw["include_trust"]
    if not isinstance(include_trust, bool):
        raise ValueError(f"include_trust must be true or false, got {include_trust!r}")

    metric = MetricCfg(**raw["metric"])
    if not 0 < metric.threshold < 1:
        raise ValueError(f"metric.threshold must be in (0, 1), got {metric.threshold}")
    if abs(metric.detection_weight + metric.agreement_weight - 1.0) > 1e-6:
        warnings.warn(f"metric weights sum to {metric.detection_weight + metric.agreement_weight}, not 1.0")

    data = _data(raw["data"])
    if data.feedback_size > 100:
        warnings.warn(f"data.feedback_size = {data.feedback_size}: GEPA degrades past ~100 feedback examples")

    raw_evals = raw.get("evals", {})
    evals = EvalsCfg(
        datasets={k: int(v) for k, v in (raw_evals.get("datasets") or {}).items()},
        fpr_points=tuple(raw_evals.get("fpr_points", (0.01, 0.05))),
    )
    dynamics = DynamicsCfg(**raw.get("dynamics", {}))

    artifact = Path(raw["artifact"])
    if not artifact.is_absolute():
        artifact = REPO_ROOT / artifact

    return Config(
        include_trust=include_trust,
        trust_history=bool(raw.get("trust_history", False)),
        summary_memory=bool(raw.get("summary_memory", False)),
        user_only=bool(raw.get("user_only", False)),
        student=_lm(raw["student"]),
        reflection=_lm(raw["reflection"]),
        gepa=GepaCfg(**raw["gepa"]),
        metric=metric,
        data=data,
        evals=evals,
        dynamics=dynamics,
        artifact=artifact,
    )
