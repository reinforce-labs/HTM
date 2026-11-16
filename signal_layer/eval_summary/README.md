`eval_summary` evaluates the quality of the signal layer's rolling summaries using an independent LLM judge. Unlike `signal_layer/evals/`, which measures detection performance, this module measures summary faithfulness and utility.

The evaluation compares GEPA-optimized summaries against the seed prompt using multiple summary-quality dimensions.

## Structure

| File | Description |
|------|-------------|
| `config.yaml` | Judge configuration, summary runs to compare, sampling settings, and output paths. |
| `sample.py` | Samples conversations for evaluation from paired summary runs. |
| `judge.py` | Runs the LLM-as-judge evaluation and writes per-conversation results. |
| `analyze.py` | Aggregates judge scores and computes summary statistics and inter-judge agreement. |
| `prompts/` | Prompt templates used by the summary-quality judge. |

## Evaluation

The judge scores each summary on six dimensions:

- Faithfulness
- Gating-fact retention
- Staleness
- Redundancy
- Sufficiency
- State recording

## Usage

First generate the summary evaluation runs:

```bash
python -m signal_layer.main --stage eval --name final_gepa_summary
python -m signal_layer.main --stage eval --name final_seed_summary --artifact seed
```

Then run the summary evaluation pipeline:

```bash
python -m signal_layer.eval_summary.main --stage sample
python -m signal_layer.eval_summary.main --stage judge
python -m signal_layer.eval_summary.main --stage analyze
```

The paths to the summary runs, judge models, and sampling configuration are specified in `config.yaml`.