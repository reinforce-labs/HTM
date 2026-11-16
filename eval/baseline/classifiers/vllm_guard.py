from __future__ import annotations

import concurrent.futures
import math

import requests
from transformers import AutoTokenizer

from eval.baseline.classifiers.base import Guard, Message

MODEL_ID = "meta-llama/Llama-Guard-3-8B"
GRANITE_MODEL_ID = "ibm-granite/granite-guardian-3.2-5b"
GRANITE_GUARDIAN_CONFIG = {"risk_name": "harm"}


def _softmax_pair(pos: float | None, neg: float | None) -> float:
    """P(pos) from the two verdict-token logprobs; None means the token was outside top-k."""
    if pos is None and neg is None:
        return float("nan")                              # neither in top-k: self_check will catch a broken readout
    if pos is None:
        return 0.0
    if neg is None:
        return 1.0
        
    m = max(pos, neg)
    ep, en = math.exp(pos - m), math.exp(neg - m)
    return ep / (ep + en)


def _p_unsafe(top_logprobs: dict[str, float]) -> float:
    """softmax over the safe/unsafe logprobs among the first token's top-k (Llama Guard)."""
    safe = unsafe = None
    for token, logprob in top_logprobs.items():
        t = token.strip().lower()
        if t == "safe" and safe is None:
            safe = logprob
        elif t in ("unsafe", "un") and unsafe is None:   # "unsafe" is one token; "un" guards a rare split
            unsafe = logprob
    return _softmax_pair(unsafe, safe)


def _p_yes(top_logprobs: dict[str, float]) -> float:
    """softmax over the Yes/No logprobs among the first token's top-k (Granite Guardian verdict)."""
    yes = no = None
    for token, logprob in top_logprobs.items():
        t = token.strip().lower()
        if t == "yes" and yes is None:
            yes = logprob
        elif t == "no" and no is None:
            no = logprob
    return _softmax_pair(yes, no)


class VLLMLlamaGuard(Guard):
    name = "llama_guard"

    def __init__(self, api_base: str, model: str = MODEL_ID, tokenizer_id: str = MODEL_ID,
                 top_logprobs: int = 20, concurrency: int = 32, timeout_s: int = 120,
                 max_prompt_tokens: int = 30000, cache: bool = True) -> None:
        super().__init__(cache=cache)
        self.url = api_base.rstrip("/") + "/completions"
        self.model = model
        self.top_logprobs = top_logprobs
        self.concurrency = concurrency
        self.timeout_s = timeout_s
        self.max_prompt_tokens = max_prompt_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.session = requests.Session()
        self.self_check()

    def _prompt(self, messages: list[Message]) -> str:
        # Llama Guard emits "\n\n" then the verdict; append it so the sampled token is safe/unsafe
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "\n\n"

    def _token_ids(self, messages: list[Message]) -> list[int]:
        """Prompt as token ids (add_special_tokens=False; the template carries BOS), left-truncated
        so the verdict-bearing template tail survives when a turn is very long."""
        ids = self.tokenizer(self._prompt(messages), add_special_tokens=False)["input_ids"]
        return ids[-self.max_prompt_tokens:] if len(ids) > self.max_prompt_tokens else ids

    def _verdict_prob(self, first_top_logprobs: dict[str, float]) -> float:
        """P(unsafe) from the first sampled token's top-k logprobs. Subclasses override to
        read a different verdict token pair (e.g. Granite's Yes/No)."""
        return _p_unsafe(first_top_logprobs)

    def _score_uncached(self, messages: list[Message]) -> float:
        payload = {
            "model": self.model,
            "prompt": self._token_ids(messages),
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": self.top_logprobs,
        }
        resp = self.session.post(self.url, json=payload, timeout=self.timeout_s)
        if resp.status_code != 200:
            raise RuntimeError(f"{resp.status_code} from {self.url}: {resp.text[:400]}")

        logprobs = resp.json()["choices"][0]["logprobs"]
        if not logprobs or not logprobs.get("top_logprobs"):
            raise RuntimeError(f"{self.url} returned no logprobs — start vLLM with logprobs support")
        return self._verdict_prob(logprobs["top_logprobs"][0])

    def score_batch(self, batch: list[list[Message]], batch_size: int = 16, desc: str | None = None) -> list[float]:
        """Fire requests concurrently; vLLM batches them server-side (batch_size is ignored here)."""
        from tqdm import tqdm

        out: list[float] = [0.0] * len(batch)
        fails = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(self.score_messages, m): i for i, m in enumerate(batch)}
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(batch), desc=desc, disable=desc is None, unit="msg"):
                try:
                    out[futures[fut]] = fut.result()
                except Exception as e:                       # one bad cell must not kill a 36k-cell run
                    fails += 1
                    if fails <= 5:
                        tqdm.write(f"[guard] cell {futures[fut]} failed, scoring 0.0 (neutral): {e}")
        if fails:
            tqdm.write(f"[guard] {fails}/{len(batch)} cells failed and were scored 0.0 (neutral)")
        return out


class VLLMGraniteGuardian(VLLMLlamaGuard):
    name = "granite"

    def __init__(self, api_base: str, model: str = GRANITE_MODEL_ID, tokenizer_id: str = GRANITE_MODEL_ID,
                 guardian_config: dict | None = None, **kwargs) -> None:
        self.guardian_config = guardian_config or GRANITE_GUARDIAN_CONFIG
        super().__init__(api_base, model=model, tokenizer_id=tokenizer_id, **kwargs)

    def _prompt(self, messages: list[Message]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, guardian_config=self.guardian_config, add_generation_prompt=True, tokenize=False)

    def _verdict_prob(self, first_top_logprobs: dict[str, float]) -> float:
        return _p_yes(first_top_logprobs)