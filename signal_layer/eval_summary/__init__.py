"""LLM-as-judge faithfulness evaluation of the rolling summaries.

Compares the GEPA-optimized vs seed scorer's per-turn summaries on the SAME held-out
conversations (paired), using an independent judge model. Three stages:

    sample   pick a stratified, class-balanced set of conversations (paired conv-ids from
             both summary runs) + their raw turns -> data/sampled.jsonl
    judge    a blind judge scores each conversation's summaries (per run) on faithfulness,
             safety coverage, risk calibration, and trajectory tracking -> data/judgments.jsonl
    analyze  paired GEPA-vs-seed stats (means, bootstrap CIs, win rate), overall + per class

    python -m signal_layer.eval_summary.main --stage sample
    python -m signal_layer.eval_summary.main --stage judge
    python -m signal_layer.eval_summary.main --stage analyze
"""
