from __future__ import annotations

import json
import os
import random
from collections.abc import Iterator
from functools import partial
from pathlib import Path

from data_gen.config import DatasetCfg
from data_gen.io import REPO_ROOT
from data_gen.schema import Conversation, Source, Turn
from .helpers import hf_rows, load_messages, pair_messages


def load_wildjailbreak(cfg: DatasetCfg) -> Iterator[Conversation]:
    """Harmful prompt and completion as a single turn."""
    for row in hf_rows(cfg.source, cfg.subset, seed=cfg.seed):
        if "harmful" not in (row.get("data_type") or ""):
            continue
        prompt = (row.get("adversarial") or row.get("vanilla") or "").strip()
        completion = (row.get("completion") or "").strip()
        if not prompt:
            continue
        turn = Turn(user=prompt, assistant=completion or None)
        yield Conversation(dataset=cfg.name, turns=[turn], is_adversarial=cfg.adversarial, source=Source(repo=cfg.source))


def load_orbench(cfg: DatasetCfg) -> Iterator[Conversation]:
    """Benign-but-sensitive single-turn prompts (over-refusal stress set)."""
    for row in hf_rows(cfg.source, cfg.subset, seed=cfg.seed):
        prompt = (row.get("prompt") or "").strip()
        if prompt:
            yield Conversation(dataset=cfg.name, turns=[Turn(user=prompt)], is_adversarial=cfg.adversarial, source=Source(repo=cfg.source))


def _mhj_turns(row: dict) -> list[Turn]:

    idxs = sorted(
        int(k.split("_", 1)[1]) for k in row
        if k.startswith("message_") and k.split("_", 1)[1].isdigit() and (row[k] or "").strip()
    )

    turns: list[Turn] = []
    for i in idxs:
        cell = row[f"message_{i}"].strip()
        try:
            msg = json.loads(cell)
            role, body = msg.get("role", "user"), (msg.get("body") or "").strip()
        except (json.JSONDecodeError, TypeError):
            role, body = "user", cell
        if body and role == "user":
            turns.append(Turn(user=body))

    return turns


def load_mhj(cfg: DatasetCfg) -> Iterator[Conversation]:
    import csv
    from huggingface_hub import hf_hub_download, list_repo_files

    rows: list[dict] = []
    for name in (f for f in list_repo_files(cfg.source, repo_type="dataset") if f.endswith(".csv")):
        path = hf_hub_download(cfg.source, name, repo_type="dataset")
        with open(path, newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    random.Random(cfg.seed).shuffle(rows)

    for row in rows:
        turns = _mhj_turns(row)
        if turns:
            yield Conversation(
                dataset=cfg.name, turns=turns, is_adversarial=cfg.adversarial,
                source=Source(repo=cfg.source, native_id=str(row.get("question_id") or "") or None),
            )


def load_safedialbench(cfg: DatasetCfg) -> Iterator[Conversation]:
    fields = ("dialogue", "conversation", "conversations", "messages", "turns")
    base = Path(os.getenv("SAFEDIALBENCH_DIR", REPO_ROOT / ".cache" / "SafeDialBench-Dataset"))
    files = sorted((base / "data" / "complete").glob("datasets_en*.jsonl"))
    if not files:
        raise FileNotFoundError(f"SafeDialBench not found under {base} — git clone it first.")
    lines = [ln for path in files for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    random.Random(cfg.seed).shuffle(lines)

    for line in lines:
        row = json.loads(line)
        field = next((f for f in fields if row.get(f)), None)
        turns = pair_messages(row.get(field)) if field else []
        if turns:
            yield Conversation(dataset=cfg.name, turns=turns, is_adversarial=cfg.adversarial, source=Source(repo=cfg.source))


LOADERS = {
    "xguard": partial(load_messages, message_field="conversations"),
    "xguard_synthetic": partial(load_messages, message_field="conversations"),
    "safedialbench": load_safedialbench,
    "wildchat": partial(load_messages, message_field="conversation", id_field="conversation_hash", streaming=True),
    "lmsys": partial(load_messages, message_field="conversation", id_field="conversation_id", streaming=True),
    "wildjailbreak": load_wildjailbreak,
    "mhj": load_mhj,
    "safemtdata": partial(load_messages, message_field="conversations", id_field="id"),
    "sharegpt": partial(load_messages, message_field="conversations", id_field="id"),
    "orbench_hard": load_orbench,
}


def get_loader(name: str):
    if name not in LOADERS:
        raise KeyError(f"no loader for '{name}'; have {sorted(LOADERS)}")
    return LOADERS[name]