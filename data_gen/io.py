from __future__ import annotations
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def raw_path(dataset: str) -> Path:
    return DATA_DIR / dataset / "raw.jsonl"


def assistant_path(dataset: str) -> Path:
    return DATA_DIR / dataset / "assistant.jsonl"


def conversation_path(dataset: str) -> Path:
    """Assistant.jsonl (victim-filled) when present, else raw.jsonl."""
    path = assistant_path(dataset)
    return path if path.exists() else raw_path(dataset)


def teacher_path(dataset: str) -> Path:
    return DATA_DIR / dataset / "teacher.jsonl"


def step_path(dataset: str) -> Path:
    return DATA_DIR / dataset / "step.jsonl"


def metadata_path() -> Path:
    return DATA_DIR / "metadata.jsonl"


def read_jsonl(path: Path | str) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def done_ids(path: Path | str, key: str = "conv_id") -> set[str]:
    """conv_ids already written to a stage file — used to resume."""
    return {rec[key] for rec in read_jsonl(path) if key in rec}


def append_jsonl(path: Path | str, records: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_jsonl(path: Path | str, records: list[dict]) -> int:
    """Overwrite a jsonl file with `records` (used to rewrite the metadata index)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def chunked(items: Iterable, size: int) -> Iterator[list]:
    buf: list = []
    for x in items:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf