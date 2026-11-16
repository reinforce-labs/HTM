# Homeostatic Trust Model

Official implementation of **A Homeostatic Framework for Measuring Trust in LLM Conversations** ([arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)).

HTM detects multi-turn jailbreaks by treating a conversation as a trajectory rather than a sequence of independent turns. The system consists of two components:

- **Signal layer** (`signal_layer/`) — a small instruction-following LLM that summarizes each conversation turn into six interpretable signals. Its prompt is optimized offline using GEPA to match trajectory-level teacher annotations.
- **Trust layer** (`htm/`) — a frozen homeostatic controller that integrates the extracted signals over time and maps them to a graded policy (e.g., allow, clarify, or refuse).

At inference time, only the small signal model is executed; teacher models are used only during offline data generation and prompt optimization.

## Repository structure

| Directory | Description |
|-----------|-------------|
| `htm/` | Homeostatic trust controller and policy logic. |
| `signal_layer/` | Signal extraction model, GEPA optimization, and evaluation. |
| `data_gen/` | Dataset acquisition and offline labeling pipeline. |
| `data/` | Generated datasets and labels (created at runtime). |
| `eval/` | Baseline guard implementations and evaluation utilities. |
| `results/` | Evaluation outputs behind the paper's results (see below). |

Each package contains its own README describing its functionality and usage.

## Installation

```bash
pip install -e ".[gepa]"
cp .env.example .env
```

See `.env.example` for required API keys and optional configuration.

## Reproducing the pipeline

### 1. Generate the datasets

Follow the instructions in `data/README.md` and `data_gen/README.md` to download, preprocess, and label the datasets.

### 2. Optimize the signal layer

```bash
python -m signal_layer.main --stage optimize
```

See `signal_layer/README.md` for optimization, evaluation, and comparison commands.

### 3. Evaluate baselines

Run the baseline detectors in `eval/` and compare them against HTM using the evaluation utilities in `signal_layer`.

## Notes

- Processed datasets and evaluation artifacts are generated locally and are not checked into the repository.
- Raw datasets are not redistributed. Refer to `data/README.md` for public sources and preprocessing details.
- Configuration is defined in:
  - `htm/config/htm.yaml`: trust controller
  - `signal_layer/config/gepa.yaml`: signal layer, optimization, and evaluation

## Released artifacts

- **Optimized signal prompt:** `signal_layer/prompts/student_optimized_summary.txt` is the GEPA-optimized prompt used for the paper's main results; `student_seed_summary.txt` is its hand-written seed.
- **Evaluation results:** `results/*.json.gz` hold the per-conversation, per-turn outputs (signals, caution, trust, control, action) of the 1.7B signal model for the 3,484 held-out conversations, keyed by conversation ID, for each memory configuration in the paper. Run `python results/reproduce_table.py` to recompute the paper's memory-ablation table from them. Conversation text and model-generated summaries are not included; regenerate them with the data pipeline.

## License

This project is released under the [Apache License 2.0](LICENSE). The datasets used in the paper are not redistributed and remain subject to their original licenses and terms of use.

## Citation

```bibtex
@article{htm2026,
  title   = {A Homeostatic Framework for Measuring Trust in LLM Conversations},
  author  = {Parigi, Paolo and Vaidya, Arjun and Yildiz, Veysi and Ren, Shiming},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```
