from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.baseline.classifiers.base import Guard, Message, batched_score

MODEL_ID = "ibm-granite/granite-guardian-3.2-5b"
_GUARDIAN_CONFIG = {"risk_name": "harm"}


class GraniteGuardian(Guard):
    name = "granite"

    def __init__(self, device: str = "cuda", cache: bool = True) -> None:
        super().__init__(cache=cache)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map=device).eval()

        self._yes_id = self.tokenizer.convert_tokens_to_ids("Yes")
        self._no_id = self.tokenizer.convert_tokens_to_ids("No")
        self.self_check()

    def _prompt(self, messages: list[Message]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, guardian_config=_GUARDIAN_CONFIG, add_generation_prompt=True, tokenize=False)

    @torch.no_grad()
    def _score_uncached(self, messages: list[Message]) -> float:
        inputs = self.tokenizer(self._prompt(messages), return_tensors="pt", add_special_tokens=False, truncation=True, max_length=4096).to(self.device)
        logits = self.model(**inputs, logits_to_keep=1).logits[0, -1]
        pair = torch.stack([logits[self._no_id], logits[self._yes_id]])
        return float(torch.softmax(pair, dim=0)[1])

    def score_batch(self, batch: list[list[Message]], batch_size: int = 16, desc: str | None = None) -> list[float]:
        prompts = [self._prompt(m) for m in batch]
        return batched_score(self.model, self.tokenizer, prompts, self._no_id, self._yes_id, self.device, batch_size, desc=desc)
