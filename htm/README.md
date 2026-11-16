The `htm` package implements the Homeostatic Trust Model (HTM): a frozen control layer that converts the signal layer's per-turn outputs into a trust trajectory and a graded policy response.

Unlike the signal layer, the controller is not optimized. During GEPA optimization, it remains fixed and serves as part of the optimization objective.

## Structure

| File | Description |
|------|-------------|
| `adapter.py` | Aggregates per-turn signals into a single caution value. |
| `trust.py` | Implements the homeostatic trust update, producing trust and control values over time. |
| `policy.py` | Maps the control value to a policy action using configurable thresholds and hysteresis. |
| `controller.py` | End-to-end HTM controller that executes the adapter, trust update, and policy for an entire conversation. |
| `schema.py` | Shared data structures for controller outputs and policy actions. |
| `config/` | Controller configuration, including signal weights, trust dynamics, and policy thresholds. |

## Controller pipeline

For each conversation turn, the controller performs:

```
Signals → Adapter → Caution → Trust Update → Control → Policy Action
```

The resulting trajectory contains the per-turn caution, trust, control, and policy action, and is used throughout optimization and evaluation.

## Design

The controller is intentionally deterministic and configuration-driven. All controller behavior is defined by `config/htm.yaml`; prompt optimization modifies only the signal layer, never the trust dynamics or policy.