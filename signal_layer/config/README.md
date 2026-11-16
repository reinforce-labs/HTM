`gepa.yaml` is the central configuration file for the signal layer. It defines the models, prompt artifact, optimization objective, dataset splits, evaluation settings, and GEPA search budget. The configuration is loaded and validated into an immutable `Config` object.

## Configuration

| Section | Purpose |
|---------|---------|
| `student` | Online signal extraction model. |
| `reflection` | Reflection model used during GEPA optimization. |
| `metric` | GEPA optimization objective and weights. |
| `data` | Dataset splits and optimization sampling. |
| `evals` | Held-out evaluation configuration. |
| `gepa` | GEPA search parameters. |
| `dynamics` | Experimental trust-dynamics optimization. |

## Signal layer options

The signal model's context is controlled by:

- `include_trust` — include the HTM trust state.
- `trust_history` — include recent trust history.
- `summary_memory` — use a rolling conversation summary.
- `user_only` — score only user turns.

The prompt used at inference is specified by `artifact`, which points to a prompt in `signal_layer/prompts/`.

## Notes

- The GEPA objective combines trajectory-level detection with agreement to the offline teacher.
- `evals` specifies the held-out evaluation datasets and reporting settings.
- The `dynamics` configuration is experimental and is not used in the published results.