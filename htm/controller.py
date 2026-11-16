from __future__ import annotations

import numpy as np

from htm.adapter import Adapter
from htm.config import HTMConfig
from htm.policy import Policy
from htm.schema import ConversationTrace, TurnTrace
from htm.trust import TrustState


class Controller:

    def __init__(self, signal_names: list[str], cfg: HTMConfig) -> None:
        self.adapter = Adapter(signal_names, cfg.adapter, cfg.caution)
        self.trust = TrustState(memory_n=cfg.trust.memory_n, u_max=cfg.trust.u_max)
        self.policy = Policy(bands=cfg.policy.bands, deescalation_margin=cfg.policy.deescalation_margin)

    def run(self, signals: np.ndarray, conv_id: str, dataset: str, is_adversarial: bool, payload_turn: int = -1) -> ConversationTrace:
        """Score one conversation's [T, S] signal matrix into a ConversationTrace."""
        self.trust.reset()
        self.policy.reset()

        cautions = self.adapter.caution(signals)

        turns = []
        for caution in cautions:
            update = self.trust.step(float(caution))
            action = self.policy.step(update.control)
            turns.append(TurnTrace.from_update(update, action))

        return ConversationTrace(conv_id=conv_id, dataset=dataset, is_adversarial=is_adversarial, payload_turn=payload_turn, turns=turns)
