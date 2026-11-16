from __future__ import annotations

import concurrent.futures
from types import SimpleNamespace

import dspy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GENERATE_TIMEOUT_S = 60


class LocalLM(dspy.BaseLM):

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 512, device: str = "auto") -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.lm = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).to(self.device).eval()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)   # serializes + bounds every call

    def _prompt(self, messages: list[dict]) -> str:
        try:
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    @torch.no_grad()
    def _generate(self, messages: list[dict], max_new_tokens: int, temperature: float) -> tuple[str, int, int]:
        enc = self.tokenizer(self._prompt(messages), return_tensors="pt", add_special_tokens=False).to(self.device)
        out = self.lm.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        completion = out[0, enc["input_ids"].shape[1]:]
        text = self.tokenizer.decode(completion, skip_special_tokens=True).strip()
        return text, int(enc["input_ids"].shape[1]), int(completion.shape[0])

    def forward(self, prompt: str | None = None, messages: list[dict] | None = None, **kwargs):
        messages = messages or [{"role": "user", "content": prompt}]
        temperature = kwargs.get("temperature", self.kwargs.get("temperature", 0.0))
        max_new_tokens = kwargs.get("max_tokens", self.kwargs.get("max_tokens", 512))

        future = self._executor.submit(self._generate, messages, max_new_tokens, temperature)
        try:
            text, prompt_tokens, completion_tokens = future.result(timeout=GENERATE_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"generation exceeded {GENERATE_TIMEOUT_S}s — the model call hung") from None

        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
            usage={"prompt_tokens": prompt_tokens,
                   "completion_tokens": completion_tokens,
                   "total_tokens": prompt_tokens + completion_tokens},
            model=self.model,
        )
