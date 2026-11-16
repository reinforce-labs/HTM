from __future__ import annotations

from tqdm import tqdm

from data_gen import io
from data_gen.config import DatasetCfg
from data_gen.judge import Judge, JudgeResult
from data_gen.prompts import PROMPT_VERSION, extract_json, normalize_level, step_messages
from data_gen.schema import Conversation

CHUNK = 200


def parse_harm(result: JudgeResult) -> float | None:
    data = extract_json(result.text) if result.ok else None
    return normalize_level(data.get("harm")) if data else None


def label_chunk(judge: Judge, convs: list[Conversation]) -> list[dict]:
    # one flat batch: every user and assistant message, scored independently
    index: list[tuple[int, int, str]] = []
    messages: list[list[dict]] = []
    for ci, conv in enumerate(convs):
        for ti, turn in enumerate(conv.turns):
            index.append((ci, ti, "user"))
            messages.append(step_messages(turn.user))
            if turn.assistant:
                index.append((ci, ti, "agent"))
                messages.append(step_messages(turn.assistant))

    results = judge.batch(messages)

    step_user: list[list] = [[None] * c.n_turns for c in convs]
    step_agent: list[list] = [[None] * c.n_turns for c in convs]
    for (ci, ti, kind), result in zip(index, results):
        target = step_user if kind == "user" else step_agent
        target[ci][ti] = parse_harm(result)

    records = []
    for i, conv in enumerate(convs):
        records.append({
            "conv_id": conv.conv_id,
            "model": judge.model,
            "prompt_version": PROMPT_VERSION,
            "step_user": step_user[i],
            "step_agent": step_agent[i],
        })
    return records


def run(ds: DatasetCfg, judge: Judge, limit: int | None = None) -> int:
    """Score each message in isolation, writing per-turn harm to step.jsonl."""
    out_file = io.step_path(ds.name)
    done = io.done_ids(out_file)

    convs = [Conversation.from_record(r) for r in io.read_jsonl(io.conversation_path(ds.name)) if r["conv_id"] not in done]
    if limit is not None:
        convs = convs[:limit]

    written = 0
    with tqdm(total=len(convs), desc=f"step {ds.name}", unit="conv") as pbar:
        for chunk in io.chunked(convs, CHUNK):
            io.append_jsonl(out_file, label_chunk(judge, chunk))
            written += len(chunk)
            pbar.update(len(chunk))
    return written