from __future__ import annotations

from tqdm import tqdm

from data_gen import io
from data_gen.config import DatasetCfg
from data_gen.judge import Judge, JudgeResult
from data_gen.prompts import PROMPT_VERSION, extract_json, normalize_level, teacher_messages
from data_gen.prompts.signals import LABELED_SIGNALS
from data_gen.schema import Conversation

# conversations labeled before each write
CHUNK = 100


def parse_turn(result: JudgeResult) -> dict:
    """Parse one teacher response into per-turn signals."""
    data = extract_json(result.text) if result.ok else None

    if data is None:
        rec = {s: None for s in LABELED_SIGNALS}
        rec.update(valid=False, raw=result.text, error=result.error, ok=result.ok)
        return rec

    rec = {s: normalize_level(data.get(s)) for s in LABELED_SIGNALS}
    rec.update(valid=True, raw=result.text)
    return rec


def label_chunk(judge: Judge, convs: list[Conversation], hint: bool = True) -> list[dict]:
    turns: list[list[dict]] = [[] for _ in convs]

    for depth in range(max(c.n_turns for c in convs)):
        active = [i for i, c in enumerate(convs) if depth < c.n_turns]
        messages = [teacher_messages(convs[i].turns[: depth + 1], hint and convs[i].is_adversarial) for i in active]
        results = judge.batch(messages)

        for i, result in zip(active, results):
            turns[i].append(parse_turn(result))

    records = []
    for i, conv in enumerate(convs):
        records.append({
            "conv_id": conv.conv_id,
            "model": judge.model,
            "prompt_version": PROMPT_VERSION,
            "turns": turns[i],
        })
    return records


def run(ds: DatasetCfg, judge: Judge, limit: int | None = None, hint: bool = True) -> int:
    """Score each conversation (assistant.jsonl if victim-filled, else raw.jsonl), writing trajectory signals to teacher.jsonl."""
    out_file = io.teacher_path(ds.name)
    done = io.done_ids(out_file)

    convs = [Conversation.from_record(r) for r in io.read_jsonl(io.conversation_path(ds.name)) if r["conv_id"] not in done]
    if limit is not None:
        convs = convs[:limit]

    written = 0
    with tqdm(total=len(convs), desc=f"teacher {ds.name}", unit="conv") as pbar:
        for chunk in io.chunked(convs, CHUNK):
            io.append_jsonl(out_file, label_chunk(judge, chunk, hint))
            written += len(chunk)
            pbar.update(len(chunk))
    return written
