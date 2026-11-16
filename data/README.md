This directory contains the processed datasets used by HTM. It is **generated at runtime** and is otherwise empty.

Running the data generation pipeline creates one directory per dataset together with a top-level `metadata.jsonl`:

```
data/
├── <dataset>/
│   ├── raw.jsonl        # normalized conversations
│   ├── assistant.jsonl  # synthesized victim replies (MHJ only)
│   ├── teacher.jsonl    # trajectory-level signal labels (train/val only)
│   └── step.jsonl       # per-turn harm labels
└── metadata.jsonl       # metadata for every conversation
```

## File descriptions

- **`raw.jsonl`** – Conversations normalized into a common format.
- **`assistant.jsonl`** – Synthesized assistant responses for user-only datasets (currently MHJ).
- **`teacher.jsonl`** – Dense trajectory-level signal annotations (0-4) used to train and evaluate the HTM signal model. Generated only for training and validation datasets.
- **`step.jsonl`** – Independent per-turn harm annotations used for single-turn evaluation.
- **`metadata.jsonl`** – One record per conversation containing dataset membership, split, conversation statistics, payload turn, and other metadata used to index the processed data.

Test datasets intentionally omit `teacher.jsonl`, since they are evaluated using only per-turn (`step`) labels.

The complete dataset specification—including sources, filtering, labeling, and train/validation splits—is defined in `data_gen/config/datasets.yaml`.

## Generating the data

```bash
python -m data_gen.main --stage acquire        --only <dataset>
python -m data_gen.main --stage add_user_turns --only mhj
python -m data_gen.main --stage teacher        --only <dataset>
python -m data_gen.main --stage step           --only <dataset>
python -m data_gen.main --stage payload        --only <dataset>
```

Omit `--only <dataset>` to process every dataset listed in `datasets.yaml`.

Gated datasets (`MHJ` and `WildJailbreak`) require `HF_TOKEN` in `.env`. `SafeDialBench` is downloaded directly from GitHub (see `.env.example`).

## Datasets

| Dataset | Source | Role | Class | Labels |
|---|---|---|---|---|
| `xguard` | `marslabucla/XGuard-Train` | train | attack | teacher+step |
| `xguard_synthetic` | `marslabucla/XGuard-Train` | train | attack | teacher+step |
| `safedialbench` | `github:drivetosouth/SafeDialBench-Dataset` | train | attack | teacher+step |
| `wildchat` | `allenai/WildChat-1M` | train | benign | teacher+step |
| `lmsys` | `lmsys/lmsys-chat-1m` | train | benign | teacher+step |
| `wildjailbreak` | `allenai/wildjailbreak` | train | attack | step |
| `mhj` | `ScaleAI/mhj` | test | attack | step |
| `safemtdata` | `SafeMTData/SafeMTData` | test | attack | step |
| `sharegpt` | `theblackcat102/sharegpt-english` | test | benign | step |
| `orbench_hard` | `bench-llm/or-bench` | test | benign | step |