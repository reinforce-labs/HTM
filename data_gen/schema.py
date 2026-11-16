from __future__ import annotations
import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass
class Turn:
    user: str
    assistant: str | None = None


@dataclass
class Source:
    """Where a conversation came from. Kept for provenance and re-fetch."""
    repo: str
    revision: str | None = None
    native_id: str | None = None


@dataclass
class Conversation:
    dataset: str
    turns: list[Turn]
    is_adversarial: bool
    source: Source | None = None
    conv_id: str = ""

    def __post_init__(self) -> None:
        if not self.conv_id:
            self.conv_id = conv_id(self.turns)

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    def to_record(self) -> dict:
        return {
            "conv_id": self.conv_id,
            "dataset": self.dataset,
            "is_adversarial": self.is_adversarial,
            "source": asdict(self.source) if self.source else None,
            "turns": [{"user": t.user, "assistant": t.assistant} for t in self.turns],
        }

    @classmethod
    def from_record(cls, rec: dict) -> "Conversation":
        turns = [Turn(user=t["user"], assistant=t.get("assistant")) for t in rec["turns"]]
        src = rec.get("source")
        return cls(
            dataset=rec["dataset"],
            turns=turns,
            is_adversarial=rec["is_adversarial"],
            source=Source(**src) if src else None,
            conv_id=rec["conv_id"],
        )


def conv_id(turns: list[Turn]) -> str:
    """Stable 16-hex content hash of a conversation (sha256 of the turn text)."""
    canon = json.dumps(
        [[t.user, t.assistant] for t in turns],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]