from __future__ import annotations

from data_gen import io
from data_gen.config import DatasetCfg

PAYLOAD_THRESHOLD = 0.75   # level 3 of the 0-4 step scale


def payload_turn(step_user: list) -> int | None:
    for i, harm in enumerate(step_user):
        if harm is not None and harm >= PAYLOAD_THRESHOLD:
            return i
    return None


def run(ds: DatasetCfg) -> int:
    """Fill payload_turn in metadata.jsonl for this dataset's adversarial conversations."""
    steps = {r["conv_id"]: r for r in io.read_jsonl(io.step_path(ds.name))}
    rows = list(io.read_jsonl(io.metadata_path()))

    updated = 0
    for row in rows:
        if row["dataset"] != ds.name or not row["is_adversarial"]:
            continue

        step = steps.get(row["conv_id"])
        if not step:
            continue

        turn = payload_turn(step["step_user"])
        if turn is not None:
            row["payload_turn"] = turn
            updated += 1

    io.write_jsonl(io.metadata_path(), rows)
    return updated
