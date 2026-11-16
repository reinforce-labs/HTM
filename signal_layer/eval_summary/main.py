from __future__ import annotations

import argparse
from pathlib import Path

import yaml

DEFAULT_CONFIG = "signal_layer/eval_summary/config.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-as-judge faithfulness eval of GEPA vs seed summaries.")
    ap.add_argument("--stage", required=True, choices=("sample", "judge", "analyze"))
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--judge", help="judge stage only: run just this judge name (see judges: in "
                                    "config); omit to run all configured judges. Useful for running "
                                    "two judges in parallel shells for wall-clock speed.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    if args.stage == "sample":
        from signal_layer.eval_summary.sample import sample
        sample(cfg)
    elif args.stage == "judge":
        from signal_layer.eval_summary.judge import judge
        judge(cfg, only=args.judge)
    else:
        from signal_layer.eval_summary.analyze import analyze
        analyze(cfg)


if __name__ == "__main__":
    main()
