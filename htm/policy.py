from __future__ import annotations

from htm.schema import Action

# action for each band level
_ACTIONS = (Action.OPEN, Action.NARROW, Action.CLARIFY, Action.REFUSE)


class Policy:

    def __init__(self, bands: tuple[float, float, float] = (0.3, 0.6, 0.9), deescalation_margin: float = 0.1) -> None:
        if not all(0 < b < 1 for b in bands):
            raise ValueError(f"Bands must be in (0, 1), got {bands}")
        if list(bands) != sorted(bands):
            raise ValueError(f"Bands must be ascending, got {bands}")
        if deescalation_margin < 0:
            raise ValueError(f"deescalation_margin must be >= 0, got {deescalation_margin}")

        self.bands = bands
        self.deescalation_margin = deescalation_margin
        self.reset()

    def reset(self) -> None:
        """Start a new conversation fully open."""
        self.level = 0

    def step(self, control: float) -> Action:
        """Map this turn's control U_t to an Action."""
        up = sum(1 for b in self.bands if control >= b)

        if up > self.level:
            self.level = up                                            # escalate freely on the rising edge
        else:
            down = sum(1 for b in self.bands if control >= b - self.deescalation_margin)
            if down < self.level:
                self.level = down                                      # de-escalate only past the margin

        return _ACTIONS[self.level]

    def run(self, controls: list[float]) -> list[Action]:
        """Reset, then map one conversation's per-turn control values to actions."""
        self.reset()
        return [self.step(c) for c in controls]