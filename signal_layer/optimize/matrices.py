from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

from signal_layer.optimize import DATA_DIR, MATRICES_PATH


def materialize(cfg, source, meta: dict | None = None, out_path: Path | str = MATRICES_PATH) -> Path:
    """Student pass over the GEPA train pool -> train_matrices.json."""
    from signal_layer.data import Corpus

    corpus = Corpus(cfg.data)
    examples = corpus.feedback() + corpus.pareto()
    conversations = []
    start = time.perf_counter()

    for ex in tqdm(examples, desc="matrices", unit="conv", dynamic_ncols=True, file=sys.stdout):
        matrix = np.asarray(source.signals(ex.dataset, ex.conv_id), dtype=np.float64)
        conversations.append({
            "conv_id": ex.conv_id,
            "dataset": ex.dataset,
            "is_adversarial": ex.is_adversarial,
            "payload_turn": ex.payload_turn,
            "matrix": [[round(float(v), 4) for v in row] for row in matrix],
        })

    payload = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_conversations": len(conversations),
        "elapsed_s": round(time.perf_counter() - start, 1),
        **(meta or {}),
        "conversations": conversations,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"[dynamics-data] {len(conversations)} conversations -> {out}")
    return out


def load(path: Path | str = MATRICES_PATH) -> list[dict]:
    """Records with numpy matrices, ready for candidate evaluation."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = data["conversations"]
    for r in records:
        r["matrix"] = np.asarray(r["matrix"], dtype=np.float64)
    return records


def split(records: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Deterministic train/val split of the matrix-set (GEPA needs both)."""
    import random
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]
