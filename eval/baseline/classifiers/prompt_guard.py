from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from eval.baseline.classifiers.base import Guard, Message

MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"


class PromptGuard(Guard):
    name = "prompt_guard"

    def __init__(self, device: str = "cuda", cache: bool = True, max_length: int = 512) -> None:
        super().__init__(cache=cache)
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device).eval()
        self._malicious_idx = self._resolve_malicious_idx()
        self.self_check()

    def _resolve_malicious_idx(self) -> int:
        from eval.baseline.classifiers.base import _BENIGN_PROBE, _HARM_PROBE

        id2label = {int(i): str(l) for i, l in self.model.config.id2label.items()}
        named = [i for i, l in id2label.items() if l.upper() == "MALICIOUS"]
        if named:
            return named[0]

        for idx in (1, 0):
            self._malicious_idx = idx
            ph, pb = self._score_uncached(_HARM_PROBE), self._score_uncached(_BENIGN_PROBE)
            if ph > 0.5 > pb:
                return idx
        raise RuntimeError(f"neither label index separates the harm/benign probes "
                           f"(id2label={id2label}) — the model or scoring logic may be broken")

    @staticmethod
    def _flatten(messages: list[Message]) -> str:
        """Prompt Guard scores raw text, not a chat template — join the turn(s) verbatim."""
        return "\n\n".join(m["content"] for m in messages if m.get("content"))

    @torch.no_grad()
    def _score_chunk(self, chunk_ids: list[int]) -> float:
        text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        logits = self.model(**enc).logits[0]
        return float(torch.softmax(logits, dim=0)[self._malicious_idx])

    def _score_text(self, text: str) -> float:
        if not text:
            return 0.0
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        budget = self.max_length - 2   # room for the tokenizer's own special tokens
        if len(ids) <= budget:
            return self._score_chunk(ids)

        # long input: the model card's own guidance is to split into segments and scan in
        # parallel — we chunk and take the max, consistent with the peak-over-turns
        # aggregation used everywhere else in this eval
        chunks = [ids[i:i + budget] for i in range(0, len(ids), budget)]
        return max(self._score_chunk(c) for c in chunks)

    def _score_uncached(self, messages: list[Message]) -> float:
        return self._score_text(self._flatten(messages))

    def score_batch(self, batch: list[list[Message]], batch_size: int = 16, desc: str | None = None) -> list[float]:
        from tqdm import tqdm
        return [self._score_uncached(m) for m in tqdm(batch, desc=desc, disable=desc is None, unit="msg")]
