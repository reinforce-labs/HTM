from __future__ import annotations

import random

import dspy

from data_gen import io
from signal_layer.config import DataCfg


class Corpus:
    """Joins raw.jsonl (conversations), 
    teacher.jsonl (per-turn signal labels), 
    metadata.jsonl and creates the split policy."""

    def __init__(self, cfg: DataCfg) -> None:
        self.cfg = cfg
        self._conversations: dict[str, dict[str, dict]] = {}   # dataset -> conv_id -> record
        self._teacher: dict[str, dict[str, list[dict]]] = {}   # dataset -> conv_id -> turn labels
        self._metadata: dict[str, list[dict]] | None = None    # dataset -> rows
        self._splits: tuple[list, list] | None = None          # (feedback, pareto), built once

    # ---- raw access -------------------------------------------------------------------

    def conversation(self, dataset: str, conv_id: str | None = None) -> dict:
        """One conversation record; the dataset's first when conv_id is omitted."""
        records = self._records(dataset)
        if not records:
            raise FileNotFoundError(f"no conversations at {io.conversation_path(dataset)}")
        if conv_id is None:
            return next(iter(records.values()))
        try:
            return records[conv_id]
        except KeyError:
            raise FileNotFoundError(f"conversation {conv_id!r} not found in {io.conversation_path(dataset)}") from None

    def metadata(self, conv_id: str) -> dict | None:
        """The metadata.jsonl row (is_adversarial, payload_turn, role) for one conversation."""
        for rows in self._rows().values():
            for row in rows:
                if row["conv_id"] == conv_id:
                    return row
        return None

    def clear(self) -> None:
        self._conversations.clear()
        self._teacher.clear()
        self._metadata = None
        self._splits = None

    # ---- example sets -----------------------------------------------------------------

    def feedback(self) -> list[dspy.Example]:
        """GEPA trainset: the examples whose failures the reflector reads."""
        return self._optimization_splits()[0]

    def pareto(self) -> list[dspy.Example]:
        """GEPA valset: candidate prompts are selected on these."""
        return self._optimization_splits()[1]

    def test(self) -> list[dspy.Example]:
        """Held-out evaluation. Datasets also used in optimization contribute val-role rows only."""
        optimization = set(self.cfg.adversarial + self.cfg.benign)
        examples = []
        for dataset in dict.fromkeys(self.cfg.test_adversarial + self.cfg.test_benign):   # ordered, deduped
            roles = ("val",) if dataset in optimization else ("train", "val")
            examples.extend(self._pool([dataset], roles=roles))
        return examples

    def eval_examples(self, caps: dict[str, int]) -> list[dspy.Example]:
        optimization = set(self.cfg.adversarial + self.cfg.benign)
        rng = random.Random(self.cfg.seed)
        examples = []
        for dataset, cap in caps.items():
            roles = ("val",) if dataset in optimization else ("train", "val", "test")
            pool = self._pool([dataset], roles=roles)
            if not pool:
                print(f"[eval] WARNING: {dataset} contributed 0 conversations "
                      f"(not acquired yet, or no rows with role in {roles}) — skipping")
                continue
            rng.shuffle(pool)
            examples.extend(pool if cap < 0 else pool[:cap])
        return examples

    # ---- internals --------------------------------------------------------------------

    def _records(self, dataset: str) -> dict[str, dict]:
        if dataset not in self._conversations:
            self._conversations[dataset] = {r["conv_id"]: r for r in io.read_jsonl(io.conversation_path(dataset))}
        return self._conversations[dataset]

    def _labels(self, dataset: str) -> dict[str, list[dict]]:
        """conv_id -> per-turn signal dicts; empty for step-only/test datasets (no teacher.jsonl)."""
        if dataset not in self._teacher:
            self._teacher[dataset] = {r["conv_id"]: r["turns"] for r in io.read_jsonl(io.teacher_path(dataset))}
        return self._teacher[dataset]

    # datasets that carry a data quirk to clean before the split (mhj: T<3 fragments)
    _SYNTH_MIN_TURNS = {"mhj": 3}

    def _rows(self) -> dict[str, list[dict]]:
        if self._metadata is None:
            by_ds: dict[str, list[dict]] = {}
            seen: set[tuple[str, str]] = set()
            for row in io.read_jsonl(io.metadata_path()):
                key = (row["dataset"], row["conv_id"])
                if key in seen:                 # dedup (mhj has duplicate metadata rows); keep first
                    continue
                seen.add(key)
                by_ds.setdefault(row["dataset"], []).append(row)
            self._apply_synth_split(by_ds)
            self._metadata = by_ds
        return self._metadata

    def _apply_synth_split(self, by_ds: dict[str, list[dict]]) -> None:
        """Deterministically re-role all-test datasets into a disjoint train/val partition,
        in memory (metadata.jsonl on disk is never touched). role=train is the GEPA held-in
        slice; role=val is served to eval (held-out); fragments below the min-turns quirk go
        to role=excluded and are served NOWHERE. Seeded by cfg.seed -> reproducible; train and
        val conv_ids are disjoint by construction, so GEPA data cannot leak into eval."""
        for dataset, n_train in (self.cfg.synth_split or {}).items():
            rows = by_ds.get(dataset, [])
            if not rows:
                print(f"[split] WARNING: synth_split names {dataset} but it has no metadata rows")
                continue
            records = self._records(dataset)
            min_turns = self._SYNTH_MIN_TURNS.get(dataset, 1)
            eligible = sorted(r["conv_id"] for r in rows
                              if len(records.get(r["conv_id"], {}).get("turns", [])) >= min_turns)
            rng = random.Random(self.cfg.seed)
            rng.shuffle(eligible)
            train, val = set(eligible[:n_train]), set(eligible[n_train:])
            for r in rows:
                cid = r["conv_id"]
                r["role"] = "train" if cid in train else "val" if cid in val else "excluded"
            frag = len(rows) - len(eligible)
            print(f"[split] {dataset}: {len(train)} train (held-in) / {len(val)} val (held-out)"
                  f" / {frag} excluded (T<{min_turns}) — disjoint")

    @staticmethod
    def _proxy_payload(teacher: list[dict] | None) -> int:
        """First turn the teacher scored harmful_intent_alignment 4/4"""
        for t, labels in enumerate(teacher or []):
            if labels.get("valid") and (labels.get("harmful_intent_alignment") or 0.0) >= 1.0:
                return t
        return -1

    @classmethod
    def _example(cls, row: dict, conv: dict, teacher: list[dict] | None) -> dspy.Example:
        payload = row.get("payload_turn")
        if payload is None and row["is_adversarial"]:
            payload = cls._proxy_payload(teacher)       # real metadata payload takes precedence
        return dspy.Example(
            conv=conv,
            conv_id=row["conv_id"],
            dataset=row["dataset"],
            is_adversarial=row["is_adversarial"],
            payload_turn=payload if payload is not None else -1,
            teacher=teacher,
        ).with_inputs("conv")

    def _pool(self, datasets: list[str] | tuple[str, ...], roles: tuple[str, ...]) -> list[dspy.Example]:
        examples = []
        for dataset in datasets:
            records, labels = self._records(dataset), self._labels(dataset)
            for row in self._rows().get(dataset, []):
                conv = records.get(row["conv_id"])
                if row["role"] not in roles or not conv or not conv["turns"]:
                    continue
                examples.append(self._example(row, conv, labels.get(row["conv_id"])))
        return examples

    @staticmethod
    def _take(adversarial: list, benign: list, n: int) -> list:
        """Pop n examples, alternating adversarial/benign (~50/50 stratification)."""
        out: list = []
        while len(out) < n and (adversarial or benign):
            pool = adversarial if (len(out) % 2 == 0 and adversarial) or not benign else benign
            out.append(pool.pop())
        return out

    def _capped_pool(self, datasets: tuple[str, ...]) -> list[dspy.Example]:
        """Train-role rows with optimization_caps applied PER DATASET (seeded), so a small
        held-in slice of each dataset survives the stratified draw instead of one large pool
        crowding out the rest."""
        rng = random.Random(self.cfg.seed)
        out = []
        for dataset in datasets:
            pool = self._pool([dataset], roles=("train",))
            cap = (self.cfg.optimization_caps or {}).get(dataset, -1)
            if 0 <= cap < len(pool):
                rng.shuffle(pool)
                pool = pool[:cap]
            if not pool:
                print(f"[optimize] WARNING: {dataset} contributed 0 train-role rows to the GEPA pool")
            out.extend(pool)
        return out

    def _optimization_splits(self) -> tuple[list, list]:
        """(feedback, pareto) — disjoint, stratified, train-role rows only, built once."""
        if self._splits is None:
            adversarial = self._capped_pool(self.cfg.adversarial)
            benign = self._capped_pool(self.cfg.benign)
            if not adversarial or not benign:
                raise RuntimeError("empty optimization pool: run the data_gen acquire/teacher stages "
                                   "for the configured datasets first")

            rng = random.Random(self.cfg.seed)
            rng.shuffle(adversarial)
            rng.shuffle(benign)

            feedback = self._take(adversarial, benign, self.cfg.feedback_size)
            pareto = self._take(adversarial, benign, self.cfg.pareto_size)
            self._splits = (feedback, pareto)
        return self._splits
