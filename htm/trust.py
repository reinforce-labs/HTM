from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TrustUpdate:
    caution: float    # C_t  net per-turn evidence; > 0 is risk, <= 0 is reassurance
    baseline: float   # B_t  n-turn memory of trust, entering this turn
    trust: float      # τ_t  trust after this turn
    control: float    # U_t  strictness fed to the policy, 1 - trust


class TrustState:

    def __init__(self, memory_n: int = 4, u_max: float = 0.9) -> None:
        if memory_n < 1:
            raise ValueError(f"memory_n must be >= 1, got {memory_n}")
        if not 0 < u_max < 1:
            raise ValueError(f"u_max must be in (0, 1), got {u_max}")

        self.memory_n = memory_n
        self.u_max = u_max
        self.reset()

    def reset(self) -> None:
        """Begin a new conversation at the guarded trust cap."""
        self.baseline = self.u_max
        self.trust = self.u_max

    def step(self, caution: float) -> TrustUpdate:
        """Advance one turn given this turn's caution C_t."""

        # update baseline (memory of past trust)
        self.baseline = (1.0 - 1.0 / self.memory_n) * self.baseline + (1.0 / self.memory_n) * self.trust

        # trust is easy to lose (exponential decay), hard to gain (additive)
        if caution > 0:
            self.trust = self.baseline * math.exp(-caution)
        else:
            self.trust = min(self.u_max, self.baseline + abs(caution))

        return TrustUpdate(caution=caution, baseline=self.baseline, trust=self.trust, control=1.0 - self.trust)

    def run(self, cautions: list[float]) -> list[TrustUpdate]:
        """Reset, then advance through one conversation's per-turn caution values."""
        self.reset()
        return [self.step(c) for c in cautions]