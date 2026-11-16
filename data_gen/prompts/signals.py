from __future__ import annotations

# negative evidence
RISK_SIGNALS = (
    "harmful_intent_alignment",
    "refusal_persistence",
    "contextual_reframing",
    "intent_concealment",
    "harmful_compliance",
)

# positive evidence
REASSURANCE_SIGNALS = ("task_legitimacy",)

# model predicted signals
LABELED_SIGNALS = RISK_SIGNALS + REASSURANCE_SIGNALS