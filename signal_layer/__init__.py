"""GEPA-optimized small-LLM signal layer.

Evolves the teacher rubric (data_gen/prompts/teacher.txt) into instructions that let a
~1.5B student model reproduce the six trajectory signals directly, as the online scorer.
The trust layer (htm/*) is frozen and sits inside the optimization objective.
See signal_layer/README.md.

One class per responsibility:
    scorer.SignalScorer          conversation -> [T, 6] matrix (owns LM, program, artifact)
    scorer.LLMSignalsSource      the SignalsSourceLike implementation used by evaluation
    objective.TrustLayerObjective  the GEPA metric: detection + agreement + feedback
    data.Corpus                  raw/teacher/metadata joins and the split policy
    pipeline.Optimization/.Evaluation  orchestration; main.py is the CLI

Only the config surface is imported here so `import signal_layer` stays dspy-free.
"""

from signal_layer.config import Config, load_config

__all__ = ["Config", "load_config"]
