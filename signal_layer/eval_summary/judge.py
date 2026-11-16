"""LLM-as-judge scoring of the rolling summaries against the full transcript."""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

METRICS = ("faithfulness", "gating_fact_retention", "staleness", "redundancy", "sufficiency", "state_vs_verdict")

_DEFAULT_PROMPT = Path(__file__).resolve().parent / "prompts" / "judge.txt"
_RETRY_STATUS = {429, 500, 502, 503, 529}


def _render(conv: dict, summaries: list[str]) -> str:
    label = "ATTACK (a known adversarial / multi-turn jailbreak attempt)" if conv["is_adversarial"] \
        else "BENIGN (a known ordinary, non-adversarial conversation)"
    lines = [f"GROUND TRUTH LABEL: {label}", "", "CONVERSATION:"]
    for i, t in enumerate(conv["turns"]):
        lines.append(f"[turn {i}] USER: {t.get('user') or ''}")
        if t.get("assistant"):
            lines.append(f"[turn {i}] ASSISTANT: {t['assistant']}")
    lines.append("\nMONITOR SUMMARIES (one after each turn):")
    for i, s in enumerate(summaries):
        lines.append(f"[after turn {i}] {s}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)   # drop reasoning blocks
    text = re.sub(r"```(?:json)?", "", text).strip()                    # drop markdown fences
    try:
        return json.loads(text)                                        # clean JSON: fast path
    except Exception:
        pass
    for start in [i for i, ch in enumerate(text) if ch == "{"]:        # else: first balanced {...} that parses
        depth = 0
        for end in range(start, len(text)):
            depth += (text[end] == "{") - (text[end] == "}")
            if depth == 0:
                try:
                    return json.loads(text[start:end + 1])
                except Exception:
                    break
    return None


def _call(judge_cfg: dict, key: str, system: str, prompt: str, max_retries: int = 4) -> str:
    url = judge_cfg["api_base"].rstrip("/") + "/chat/completions"
    payload = {
        "model": judge_cfg["model"],
        "temperature": judge_cfg["temperature"],
        "max_tokens": judge_cfg["max_tokens"],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        **(judge_cfg.get("extra_body") or {}),        # e.g. chat_template_kwargs to disable thinking
    }
    backoff = 2.0
    for attempt in range(max_retries):
        resp = requests.post(url, json=payload, headers={"Authorization": f"Bearer {key}"},
                             timeout=judge_cfg["timeout_s"])
        if resp.status_code == 200:
            msg = resp.json()["choices"][0]["message"]
            return msg.get("content") or msg.get("reasoning_content") or ""
        if resp.status_code in _RETRY_STATUS and attempt < max_retries - 1:
            time.sleep(backoff)
            backoff *= 2
            continue
        raise RuntimeError(f"{resp.status_code}: {resp.text[:300]}")
    raise RuntimeError("exhausted retries")   # unreachable, satisfies type checkers


def _run_one_judge(judge_cfg: dict, system: str, convs: list[dict], out_path: Path) -> None:
    key = os.environ.get(judge_cfg["api_key_env"])
    if not key:
        raise SystemExit(f"set {judge_cfg['api_key_env']} in the environment")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for l in out_path.read_text(encoding="utf-8").split("\n"):
            if l.strip():
                r = json.loads(l)
                done.add((r["conv_id"], r["run"]))

    jobs = [(c, run) for c in convs for run in ("gepa", "seed") if (c["conv_id"], run) not in done]
    name = judge_cfg["name"]
    print(f"[judge:{name}] model={judge_cfg['model']}  concurrency={judge_cfg['concurrency']}  "
          f"to judge: {len(jobs)}  (cached: {len(done)})")
    if not jobs:
        return

    def work(item):
        conv, run = item
        base = {"conv_id": conv["conv_id"], "dataset": conv["dataset"],
                "is_adversarial": conv["is_adversarial"], "run": run}
        try:
            verdict = _extract_json(_call(judge_cfg, key, system, _render(conv, conv["summaries"][run])))
            if verdict is None:
                raise ValueError("no JSON in response")
            base.update({k: verdict.get(k) for k in (*METRICS, "hallucinations", "notes")})
        except Exception as e:
            base["error"] = str(e)[:200]
        return base

    with out_path.open("a", encoding="utf-8") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=judge_cfg["concurrency"]) as pool:
            futures = [pool.submit(work, j) for j in jobs]
            fails = 0
            bar = tqdm(concurrent.futures.as_completed(futures), total=len(jobs), desc=f"judge:{name}", unit="conv")
            for fut in bar:
                rec = fut.result()
                fails += "error" in rec
                bar.set_postfix(fails=fails)          # live signal: a spike means rate-limiting or bad JSON
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

    print(f"[judge:{name}] done -> {out_path}" + (f"  ({fails} errors, resume to retry)" if fails else ""))


def judge(cfg: dict, only: str | None = None) -> None:
    system = Path(cfg.get("prompt") or _DEFAULT_PROMPT).read_text(encoding="utf-8")
    convs = [json.loads(l) for l in Path(cfg["paths"]["sampled"]).read_text(encoding="utf-8").split("\n") if l.strip()]
    judgments_dir = Path(cfg["paths"]["judgments_dir"])

    judges = cfg["judges"]
    if only:
        judges = [j for j in judges if j["name"] == only]
        if not judges:
            raise SystemExit(f"no judge named {only!r} in config (have: {[j['name'] for j in cfg['judges']]})")

    for jc in judges:
        _run_one_judge(jc, system, convs, judgments_dir / f"{jc['name']}.jsonl")
