from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from htm.schema import ConversationTrace

# model: {"current": [T], "history": [T]}
Scores = dict[str, dict[str, list[float]]] 


def _mark_payload(ax, payload_turn: int) -> None:
    if payload_turn is not None and payload_turn >= 0:
        ax.axvline(payload_turn, color="red", lw=1.0, ls=":")


def guard_panels(scores: Scores, payload_turn: int, out_path: Path, title: str = "") -> Path:
    """Two panels, current-turn risk and history risk, one line per guard."""
    n_turns = len(next(iter(scores.values()))["current"])
    turns = np.arange(n_turns)

    fig, (ax_cur, ax_hist) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for model, series in scores.items():
        ax_cur.plot(turns, series["current"], marker="o", ms=3, label=model)
        ax_hist.plot(turns, series["history"], marker="o", ms=3, label=model)

    ax_cur.set_ylabel("P(unsafe)\ncurrent turn")
    ax_hist.set_ylabel("P(unsafe)\nhistory $\\leq t$")
    ax_hist.set_xlabel("turn")
    ax_cur.set_title(title, fontsize=9)
    for ax in (ax_cur, ax_hist):
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
        _mark_payload(ax, payload_turn)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def comparison(trace: ConversationTrace, scores: Scores, out_path: Path) -> Path:
    """Overlay HTM control U against each guard's current/history risk."""
    turns = np.arange(len(trace.turns))
    control = [t.control for t in trace.turns]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(turns, control, marker="o", ms=4, lw=2, color="black", label="HTM control $U$")
    for model, series in scores.items():
        ax.plot(turns, series["history"], marker="o", ms=3, ls="--", label=f"{model} (history)")
        ax.plot(turns, series["current"], marker="x", ms=4, ls=":", label=f"{model} (current)")

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("turn")
    ax.set_ylabel("risk / control")
    ax.set_title(f"{trace.dataset}  {trace.conv_id}  adversarial={trace.is_adversarial}  "
                 f"payload_turn={trace.payload_turn}", fontsize=9)
    ax.legend(fontsize=7)
    _mark_payload(ax, trace.payload_turn)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return out_path
