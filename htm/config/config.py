from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "htm.yaml"

REDUCERS = ("max", "mean")


@dataclass
class AdapterCfg:
    risk_step: list[str]                                    # per-turn signals, fed as level (e-)
    risk_trajectory: list[str]                              # cumulative signals, fed as turn-over-turn change (e-)
    reassurance: list[str]                                  # positive evidence (e+)
    reduce: str = "max"                                     # how multiple risk signals combine into e- (max | mean)


@dataclass
class CautionCfg:
    alpha: float = 1.0                                      # weight on risk evidence (e-)
    beta: float = 0.5                                       # weight on reassurance (e+) (alpha > beta = negativity bias)


@dataclass
class TrustCfg:
    memory_n: int = 4                                       # mirrors TrustState(memory_n=...)
    u_max: float = 0.9                                      # mirrors TrustState(u_max=...)


@dataclass
class PolicyCfg:
    bands: tuple[float, float, float] = (0.3, 0.6, 0.9)     # mirrors Policy(bands=...)
    deescalation_margin: float = 0.1                        # mirrors Policy(deescalation_margin=...)


@dataclass
class HTMConfig:
    adapter: AdapterCfg
    caution: CautionCfg
    trust: TrustCfg
    policy: PolicyCfg


def load_config(path: Path | str = CONFIG_PATH) -> HTMConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    adapter = AdapterCfg(
        risk_step=list(raw["adapter"]["risk_step"]),
        risk_trajectory=list(raw["adapter"]["risk_trajectory"]),
        reassurance=list(raw["adapter"]["reassurance"]),
        reduce=raw["adapter"].get("reduce", "max"),
    )

    if adapter.reduce not in REDUCERS:
        raise ValueError(f"adapter.reduce must be one of {REDUCERS}, got {adapter.reduce!r}")
    if not (adapter.risk_step or adapter.risk_trajectory):
        raise ValueError("adapter needs at least one risk signal (risk_step or risk_trajectory)")
    if not adapter.reassurance:
        raise ValueError("adapter.reassurance must list at least one signal")

    caution = CautionCfg(alpha=raw["caution"]["alpha"], beta=raw["caution"]["beta"])
    if caution.alpha <= caution.beta:
        warnings.warn(f"caution.alpha ({caution.alpha}) <= beta ({caution.beta}): reassurance outweighs "
                      "risk, against the HTM negativity-bias premise")

    trust = TrustCfg(memory_n=raw["trust"]["memory_n"], u_max=raw["trust"]["u_max"])
    if trust.memory_n < 1:
        raise ValueError(f"trust.memory_n must be >= 1, got {trust.memory_n}")
    if not 0 < trust.u_max < 1:
        raise ValueError(f"trust.u_max must be in (0, 1), got {trust.u_max}")

    bands = tuple(raw["policy"]["bands"])
    if len(bands) != 3 or list(bands) != sorted(bands) or not all(0 < b < 1 for b in bands):
        raise ValueError(f"policy.bands must be 3 ascending thresholds in (0, 1), got {bands}")
        
    margin = raw["policy"].get("deescalation_margin", 0.1)
    if margin < 0:
        raise ValueError(f"policy.deescalation_margin must be >= 0, got {margin}")
    policy = PolicyCfg(bands=bands, deescalation_margin=margin)

    return HTMConfig(adapter=adapter, caution=caution, trust=trust, policy=policy)