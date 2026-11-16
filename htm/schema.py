from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from htm.trust import TrustUpdate


class Action(Enum):
    OPEN = "open"          # answer normally
    NARROW = "narrow"      # answer, but scoped / safeguarded
    CLARIFY = "clarify"    # ask the user to clarify intent before proceeding
    REFUSE = "refuse"      # decline


@dataclass
class TurnTrace:
    """One turn's full HTM record."""
    caution: float
    baseline: float
    trust: float
    control: float
    action: Action

    @classmethod
    def from_update(cls, update: TrustUpdate, action: Action) -> "TurnTrace":
        return cls(update.caution, update.baseline, update.trust, update.control, action)


@dataclass
class ConversationTrace:
    """One conversation run through the HTM control layer."""
    conv_id: str
    dataset: str
    is_adversarial: bool
    payload_turn: int       # -1 when there is no labeled payload turn (e.g. benign)
    turns: list[TurnTrace]