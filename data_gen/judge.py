from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import requests

from .io import REPO_ROOT

Message = dict[str, str]

BASE_URL = "https://api.deepinfra.com/v1/openai"
TIMEOUT = 120
MAX_RETRIES = 3
_RETRY_CODES = (429, 500, 502, 503, 504)


def _load_dotenv() -> None:
    env = REPO_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class JudgeResult:
    text: str
    ok: bool
    error: str | None = None


class Judge:
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 512, concurrency: int = 48, json_mode: bool = True) -> None:
        _load_dotenv()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.concurrency = concurrency
        self.json_mode = json_mode
        
        self.api_key = os.getenv("DEEPINFRA_API_KEY", "")
        if not self.api_key:
            raise ValueError("DEEPINFRA_API_KEY is not set (add it to .env or the environment).")

        self.url = f"{os.getenv('JUDGE_BASE_URL', BASE_URL).rstrip('/')}/chat/completions"
        self._headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        # one requests.Session per thread; Session is not thread-safe
        self._local = threading.local()

    def _session(self) -> requests.Session:
        """Return this thread's requests.Session, creating it on first use."""
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(self._headers)
            self._local.session = session
        return session

    def _payload(self, messages: list[Message]) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def complete(self, messages: list[Message]) -> JudgeResult:
        """One completion. Returns ok=False after retries instead of raising."""
        last_err = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self._session().post(self.url, json=self._payload(messages), timeout=TIMEOUT)
                if resp.status_code == 200:
                    choices = resp.json().get("choices", [])
                    text = choices[0]["message"]["content"].strip() if choices and choices[0].get("message") else ""
                    return JudgeResult(text=text, ok=True)
                last_err = f"{resp.status_code}: {resp.text[:200]}"
                if resp.status_code not in _RETRY_CODES:
                    break
            except requests.RequestException as exc:
                last_err = str(exc)
            time.sleep(2 ** attempt)
        return JudgeResult(text="", ok=False, error=last_err)

    def batch(self, batch_messages: list[list[Message]]) -> list[JudgeResult]:
        """Concurrent completions, order preserved. Per-item failures isolated to that item."""
        if not batch_messages:
            return []
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(self.complete, batch_messages))