from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "datasets.yaml"


@dataclass
class JudgeCfg:
    model: str
    temperature: float = 0.0
    concurrency: int = 48


@dataclass
class VictimCfg:
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    concurrency: int = 32
    system_prompt: str = "You are a helpful AI Assistant!"


@dataclass
class DatasetCfg:
    name: str
    source: str
    role: str                                                # train | test
    count: int
    label: str                                               # teacher + step | step
    adversarial: bool
    min_turns: int = 1
    subset: str | None = None
    split: str = "train"                                     # HF split to read
    language: str | None = None
    seed: int = 0
    val_fraction: float = 0.0
    add_user_turns: bool = False
    dedup_against: list[str] = field(default_factory=list)   # skip conv_ids already acquired by these datasets

    @property
    def use_teacher(self) -> bool:
        return "teacher" in self.label


@dataclass
class Config:
    seed: int
    val_fraction: float
    prompt_version: str
    teacher: JudgeCfg
    step: JudgeCfg
    victim: VictimCfg
    datasets: dict[str, DatasetCfg]

    def names(self, role: str | None = None) -> list[str]:
        return [n for n, d in self.datasets.items() if role is None or d.role == role]


def load_config(path: Path | str = CONFIG_PATH) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    judges = raw["judges"]
    seed = raw.get("seed", 0)
    val_fraction = raw.get("val_fraction", 0.1)
    if val_fraction < 0 or val_fraction >= 1:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    datasets = {
        name: DatasetCfg(
            name=name,
            source=d["source"],
            role=d["role"],
            count=d["count"],
            label=d["label"],
            adversarial=d["adversarial"],
            min_turns=d.get("min_turns", 1),
            subset=d.get("subset"),
            split=d.get("split", "train"),
            language=d.get("language"),
            seed=seed,
            val_fraction=val_fraction,
            add_user_turns=d.get("add_user_turns", False),
            dedup_against=d.get("dedup_against", []),
        )
        for name, d in raw["datasets"].items()
    }
    return Config(
        seed=seed,
        val_fraction=val_fraction,
        prompt_version=raw.get("prompt_version", "v1"),
        teacher=JudgeCfg(**judges["teacher"]),
        step=JudgeCfg(**judges["step"]),
        victim=VictimCfg(**judges["victim"]),
        datasets=datasets,
    )