from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from data_gen import io
from data_gen.config import DatasetCfg
from data_gen.loader import get_loader
from data_gen.schema import Conversation

# buffer this many conversations before writing them to disk
FLUSH_LIMIT = 500   


def split_role(conv_id: str, ds: DatasetCfg) -> str:
    """Every 1/val_fraction-th conversation (by content hash) is held out for validation."""
    if ds.role != "train" or ds.val_fraction <= 0:
        return ds.role
    if not 0 < ds.val_fraction < 1:
        raise ValueError(f"val_fraction must be in (0, 1), got {ds.val_fraction}")
    every_nth = round(1 / ds.val_fraction)
    return "val" if int(conv_id, 16) % every_nth == 0 else "train"


def conversation_metadata(conv: Conversation, ds: DatasetCfg) -> dict:
    return {
        "conv_id": conv.conv_id,
        "dataset": ds.name,
        "role": split_role(conv.conv_id, ds),
        "is_adversarial": conv.is_adversarial,
        "n_turns": conv.n_turns,
        "payload_turn": None,
    }


def flush_buffer(raw_file: Path, metadata_file: Path, pending_raw: list[dict], pending_metadata: list[dict]) -> int:
    """Append the buffered conversations and metadata rows to disk. Also clear the buffers."""
    if not pending_raw:
        return 0

    io.append_jsonl(raw_file, pending_raw)
    io.append_jsonl(metadata_file, pending_metadata)

    written = len(pending_raw)

    pending_raw.clear()
    pending_metadata.clear()

    return written


def run(ds: DatasetCfg, limit: int | None = None) -> int:
    '''Load conversations to raw.jsonl + metadata.'''
    target_count = limit or ds.count
    raw_file, metadata_file = io.raw_path(ds.name), io.metadata_path()
    seen_ids = io.done_ids(raw_file)

    # conv_ids already acquired by other datasets
    exclude_ids: set[str] = set()
    for other in ds.dedup_against:
        exclude_ids |= io.done_ids(io.raw_path(other))

    pending_raw: list[dict] = []
    pending_metadata: list[dict] = []
    added = 0

    pbar = tqdm(total=target_count, initial=len(seen_ids), desc=f"acquire {ds.name}", unit="conv")
    for conv in get_loader(ds.name)(ds):
        if len(seen_ids) >= target_count:
            break
            
        # Skip if not long enough, already seen, or claimed by a dedup_against dataset
        if conv.n_turns < ds.min_turns or conv.conv_id in seen_ids or conv.conv_id in exclude_ids:
            continue

        seen_ids.add(conv.conv_id)
        pending_raw.append(conv.to_record())
        pending_metadata.append(conversation_metadata(conv, ds))
        pbar.update(1)

        if len(pending_raw) >= FLUSH_LIMIT:
            added += flush_buffer(raw_file, metadata_file, pending_raw, pending_metadata)

    added += flush_buffer(raw_file, metadata_file, pending_raw, pending_metadata)
    pbar.close()
    return added


def print_samples(ds: DatasetCfg, n: int = 1) -> None:
    """Print sample conversations to stdout."""

    for i, conv in enumerate(get_loader(ds.name)(ds)):
        if i >= n:
            break

        print(f"[{ds.name}] id={conv.conv_id} turns={conv.n_turns} adversarial={conv.is_adversarial}")

        for t in conv.turns[:3]:
            print("  user:", (t.user or "")[:120])
            print("  asst:", t.assistant[:120] if t.assistant else "(none)")

        print()