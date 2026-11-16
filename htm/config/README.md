# config

`htm.yaml` is the single spec the HTM control layer reads: the adapter that turns the signal
layer's per-turn signals into caution evidence, the caution weights, the trust recurrence, and
the policy bands. Declaring the run once here keeps it out of the control code.

The four blocks map onto the `C -> tau -> U -> action` loop:

- `adapter` — which signals become negative evidence `e-` and positive evidence `e+`, and how
  multiple risk signals combine (`reduce`). `risk_step` signals are fed as their level; the
  cumulative `risk_trajectory` signals are fed as their turn-over-turn change, so the trust state
  doesn't re-integrate history the signal layer already accumulated.
- `caution` — `alpha`/`beta` in `C_t = alpha*e- - beta*e+`. `alpha > beta` is the negativity bias
  (trust is quick to lose, slow to gain); `load_config` warns if that ordering is broken.
- `trust` — `memory_n` and `u_max` for the recurrence in `htm/trust.py`. The field names match the
  `TrustState` constructor, so the controller builds it directly from this block.
- `policy` — `bands` and `deescalation_margin` for `htm/policy.py`, likewise matching the `Policy`
  constructor.

`load_config` validates ranges (`u_max` in `(0,1)`, ascending bands, non-negative margin, known
`reduce`). It does **not** check the `adapter` signal names against the signal layer's actual
output — that membership check belongs in `adapter.py` at runtime, where the signal layer's
signal list is the source of truth.
