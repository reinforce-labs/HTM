from __future__ import annotations
import numpy as np
from htm.config import AdapterCfg, CautionCfg


class Adapter:

    def __init__(self, signal_names: list[str], adapter_cfg: AdapterCfg, caution_cfg: CautionCfg) -> None:
        self.names = list(signal_names)
        self._index = {name: i for i, name in enumerate(self.names)}

        # resolve configured names to columns
        self.step_cols = self._columns(adapter_cfg.risk_step)
        self.trajectory_cols = self._columns(adapter_cfg.risk_trajectory)
        self.reassurance_cols = self._columns(adapter_cfg.reassurance)

        self.reduce = adapter_cfg.reduce
        self.alpha = caution_cfg.alpha
        self.beta = caution_cfg.beta

    def _columns(self, wanted: list[str]) -> list[int]:
        unknown = [s for s in wanted if s not in self._index]
        if unknown:
            raise ValueError(f"adapter signals {unknown} are not among the signal layer's signals {self.names}")
        return [self._index[s] for s in wanted]

    def _reduce(self, mat: np.ndarray) -> np.ndarray:
        """Combine several evidence columns [T, k] into one [T] series. (max | mean)"""
        return mat.max(axis=1) if self.reduce == "max" else mat.mean(axis=1)

    def caution(self, signals: np.ndarray) -> np.ndarray:
        """[T, S] signal probabilities in [0, 1] with caution C_t = alpha*e_minus - beta*e_plus"""
        signals = np.asarray(signals, dtype=np.float64)

        e_minus = self._reduce(signals[:, self.step_cols + self.trajectory_cols])
        e_plus = self._reduce(signals[:, self.reassurance_cols])

        return self.alpha * e_minus - self.beta * e_plus