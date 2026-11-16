from __future__ import annotations

import argparse

from data_gen.config import load_config
from data_gen.judge import Judge
from data_gen.stages import acquire, add_user_turns, payload, step, teacher

STAGES = ("print_samples", "acquire", "add_user_turns", "teacher", "step", "payload")
ALL_STAGES = ("acquire", "add_user_turns", "teacher", "step", "payload")


def run_stage(stage: str, names: list[str], cfg, limit: int | None, teacher_judge, step_judge, victim_judge, hint: bool = True) -> None:
    for name in names:
        ds = cfg.datasets[name]

        if stage == "print_samples":
            acquire.print_samples(ds)
        elif stage == "acquire":
            print(f"[acquire] {name}: +{acquire.run(ds, limit)}")
        elif stage == "add_user_turns":
            if not ds.add_user_turns:
                print(f"[add_user_turns] {name}: skip (add_user_turns: false in datasets.yaml)")
            else:
                print(f"[add_user_turns] {name}: +{add_user_turns.run(ds, victim_judge, cfg.victim.system_prompt, limit)}")
        elif stage == "teacher" and ds.use_teacher:
            print(f"[teacher] {name}: +{teacher.run(ds, teacher_judge, limit, hint)}")
        elif stage == "step":
            print(f"[step] {name}: +{step.run(ds, step_judge, limit)}")
        elif stage == "payload" and ds.adversarial:
            print(f"[payload] {name}: {payload.run(ds)} payload turns")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a data-generation stage.")
    ap.add_argument("--stage", default="acquire", choices=STAGES + ("all",))
    ap.add_argument("--only", help="comma-separated dataset names (default: all)")
    ap.add_argument("--limit", type=int, help="cap conversations per dataset")
    ap.add_argument("--no-hint", action="store_true",
                    help="teacher: label adversarial convs WITHOUT the attack hint")
    args = ap.parse_args()

    cfg = load_config()
    names = args.only.split(",") if args.only else list(cfg.datasets)
    stages = list(ALL_STAGES) if args.stage == "all" else [args.stage]

    teacher_judge = Judge(cfg.teacher.model, temperature=cfg.teacher.temperature, concurrency=cfg.teacher.concurrency) if "teacher" in stages else None
    step_judge = Judge(cfg.step.model, temperature=cfg.step.temperature, concurrency=cfg.step.concurrency) if "step" in stages else None
    victim_judge = None
    if "add_user_turns" in stages:
        v = cfg.victim
        victim_judge = Judge(v.model, temperature=v.temperature, max_tokens=v.max_tokens, concurrency=v.concurrency, json_mode=False)

    for stage in stages:
        run_stage(stage, names, cfg, args.limit, teacher_judge, step_judge, victim_judge, hint=not args.no_hint)


if __name__ == "__main__":
    main()
