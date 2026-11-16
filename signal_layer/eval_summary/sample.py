"""Sampling of conversations for the summary judge --- paired on conv_id across both runs.

mode: full   -> every conv_id present in BOTH summary runs, for every dataset (no cap).
mode: capped -> a seeded, stratified, class-balanced per_dataset draw (the original cheap design).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from data_gen import io as data_io


def _load_run(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {c["conv_id"]: c for c in data["conversations"]}


def _turns_by_id(dataset: str) -> dict[str, list[dict]]:
    out = {}
    for rec in data_io.read_jsonl(data_io.conversation_path(dataset)):
        out[rec["conv_id"]] = [{"user": t.get("user"), "assistant": t.get("assistant")} for t in rec["turns"]]
    return out


def _clip(text: str | None, n: int) -> str:
    if not text:
        return text or ""
    return text if len(text) <= n else text[:n].rstrip() + " …[truncated]"


def sample(cfg: dict) -> None:
    gepa = _load_run(cfg["runs"]["gepa"])
    seed_run = _load_run(cfg["runs"]["seed"])
    common = set(gepa) & set(seed_run)                       # paired: only conv-ids in both runs

    rng = random.Random(cfg["sampling"]["seed"])
    max_chars = cfg["sampling"]["max_chars_per_message"]
    mode = cfg["sampling"].get("mode", "capped")

    by_ds: dict[str, list[str]] = {}
    for cid in common:
        by_ds.setdefault(gepa[cid]["dataset"], []).append(cid)

    datasets = sorted(by_ds) if mode == "full" else list(cfg["sampling"]["per_dataset"])

    records = []
    for dataset in datasets:
        ids = sorted(by_ds.get(dataset, []))
        if mode == "full":
            chosen = ids                                     # every paired conversation, no cap
        else:
            n = cfg["sampling"]["per_dataset"][dataset]
            if len(ids) < n:
                print(f"[sample] WARNING: {dataset} has {len(ids)} paired convs, requested {n} — taking all")
            rng.shuffle(ids)
            chosen = ids[:n]

        turns_by = _turns_by_id(dataset)
        for cid in chosen:
            turns = [{"user": _clip(t["user"], max_chars), "assistant": _clip(t["assistant"], max_chars)}
                     for t in turns_by.get(cid, [])]
            records.append({
                "conv_id": cid,
                "dataset": dataset,
                "is_adversarial": bool(gepa[cid]["is_adversarial"]),
                "turns": turns,
                "summaries": {"gepa": gepa[cid].get("memory", []), "seed": seed_run[cid].get("memory", [])},
            })

    out = Path(cfg["paths"]["sampled"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_atk = sum(r["is_adversarial"] for r in records)
    counts = {ds: sum(1 for r in records if r["dataset"] == ds) for ds in datasets}
    print(f"[sample] mode={mode}  per-dataset: {counts}")
    print(f"[sample] wrote {len(records)} convs ({n_atk} attack / {len(records) - n_atk} benign) -> {out}")
