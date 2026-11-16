This package contains the external safety guard baselines evaluated against HTM, including **Llama Guard 3**, **Llama Prompt Guard 2**, and **Granite Guardian**. Each guard produces a per-turn safety score that is evaluated using the same held-out datasets and metrics as HTM.

## Structure

| File | Description |
|------|-------------|
| `classifiers/` | Implementations of the supported safety guards. |
| `classifiers/base.py` | Common interface, batching, caching, and model validation utilities. |
| `guard_scores.py` | Runs each guard over conversations and supports multiple context variants (current turn, history, user history, etc.). |
| `device.py` | Device selection utilities. |
| `plot.py` | Visualization utilities for comparing guard scores with HTM trajectories. |

## Supported guards

- **Llama Guard 3**
- **Llama Prompt Guard 2**
- **Granite Guardian**
- Optional vLLM-backed implementations for high-throughput evaluation

## Running a baseline

Baseline results are generated through the shared evaluation pipeline so they can be compared directly with HTM.

```bash
python -m signal_layer.evals.run_guard --guard llama_guard --variant current

python -m signal_layer.evals.run_guard \
    --guard llama_guard \
    --variant history \
    --name llama_guard_history

python -m signal_layer.main --stage compare
```

Each run produces the same evaluation artifact format as HTM, enabling direct comparison of detection performance, lead time, latency, and other metrics.