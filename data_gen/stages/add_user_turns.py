from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from data_gen import io
from data_gen.config import DatasetCfg
from data_gen.judge import Judge, Message
from data_gen.schema import Conversation


def fill_conversation(victim: Judge, conv: Conversation, system_prompt: str) -> dict | None:
    """User turns through the victim model."""
    messages: list[Message] = [{"role": "system", "content": system_prompt}]
    out_turns: list[dict] = []

    for turn in conv.turns:
        messages.append({"role": "user", "content": turn.user or ""})
        result = victim.complete(messages)
        if not result.ok or not result.text:
            return None

        messages.append({"role": "assistant", "content": result.text})
        out_turns.append({"user": turn.user, "assistant": result.text})

    record = conv.to_record()
    record["turns"] = out_turns
    record["victim_model"] = victim.model
    record["victim_system_prompt"] = system_prompt
    record["victim_temperature"] = victim.temperature
    return record


def run(ds: DatasetCfg, victim: Judge, system_prompt: str, limit: int | None = None) -> int:
    """Add assistant replies in raw.jsonl and write them to assistant.jsonl."""
    out_file = io.assistant_path(ds.name)
    done = io.done_ids(out_file)

    convs = [Conversation.from_record(r) for r in io.read_jsonl(io.raw_path(ds.name)) if r["conv_id"] not in done]
    if limit is not None:
        convs = convs[:limit]

    if not convs:
        return 0

    written = 0
    with tqdm(total=len(convs), desc=f"add_user_turns {ds.name}", unit="conv") as pbar:
        with ThreadPoolExecutor(max_workers=victim.concurrency) as pool:
            for record in pool.map(lambda c: fill_conversation(victim, c, system_prompt), convs):
                if record is not None:
                    io.append_jsonl(out_file, [record])
                    written += 1
                pbar.update(1)

    return written