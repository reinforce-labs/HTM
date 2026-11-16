from __future__ import annotations

from collections.abc import Iterator

from data_gen.config import DatasetCfg
from data_gen.schema import Conversation, Source, Turn


def pair_messages(messages: list[dict] | None) -> list[Turn]:
    """Converts a message list into [Turn(user, assistant)]."""
    turns: list[Turn] = []

    for msg in messages or []:
        role = (msg.get("role") or msg.get("from") or msg.get("user") or "").lower()
        content = (msg.get("content") or msg.get("value") or msg.get("text") or "").strip()

        if not content:
            continue
        if role in ("user", "human"):
            turns.append(Turn(user=content))
        elif role in ("assistant", "gpt", "model") and turns and turns[-1].assistant is None:
            turns[-1].assistant = content

    return turns


def hf_rows(source: str, subset: str | None = None, streaming: bool = False, seed: int = 0, split: str = "train", shuffle_buffer: int = 50_000) -> Iterator[dict]:
    """Iterate a seed-shuffled split of a HuggingFace dataset."""
    from datasets import load_dataset

    ds = load_dataset(source, subset, split=split, streaming=streaming)
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer) if streaming else ds.shuffle(seed=seed)
    yield from ds


def load_messages(cfg: DatasetCfg, message_field: str, id_field: str | None = None, streaming: bool = False) -> Iterator[Conversation]:
    '''Load HF rows where the dialogue is a message list in message_field.'''

    for row in hf_rows(cfg.source, cfg.subset, streaming, cfg.seed, cfg.split):
        if cfg.language and (row.get("language") or "").lower() != cfg.language.lower():
            continue

        turns = pair_messages(row.get(message_field))
        if not turns:
            continue

        native = str(row[id_field]) if id_field and row.get(id_field) is not None else None

        yield Conversation(
            dataset=cfg.name, turns=turns, is_adversarial=cfg.adversarial,
            source=Source(repo=cfg.source, native_id=native),
        )