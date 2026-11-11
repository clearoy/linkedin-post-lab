"""Small DeepSeek chat client with JSON output, retries, and a disk cache.

Settings come from `.env` at the project root (see `.env.example`):
DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_BASE_URL.
"""

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .paths import PROCESSED, ROOT

DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_BASE_URL = "https://api.deepseek.com"
CACHE_DIR = PROCESSED / "llm_cache"


@dataclass(frozen=True)
class LLMSettings:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL

    @classmethod
    def from_env(cls, env_file: Path = ROOT / ".env", model: str | None = None) -> "LLMSettings":
        load_dotenv(env_file)
        key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not key:
            raise RuntimeError(f"DEEPSEEK_API_KEY is empty. Put your key in {env_file}.")
        return cls(
            api_key=key,
            model=model or os.getenv("DEEPSEEK_MODEL") or DEFAULT_MODEL,
            base_url=os.getenv("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL,
        )


def parse_json(text: str) -> dict:
    """Parse a JSON object from a model reply, tolerating ```json fences and stray prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, cached: bool, prompt: int = 0, completion: int = 0) -> None:
        with self._lock:
            self.calls += 1
            self.cached += cached
            self.prompt_tokens += prompt
            self.completion_tokens += completion

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class LLMClient:
    """Chat completions that return parsed JSON.

    Identical requests are served from `cache_dir`, so re-running a script
    costs nothing and gives the same result.
    """

    def __init__(self, settings: LLMSettings, cache_dir: Path | None = CACHE_DIR, max_retries: int = 4):
        from openai import OpenAI

        self.settings = settings
        self.client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.usage = Usage()

    def _cache_path(self, request: dict) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def chat_json(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 8000) -> dict:
        request = {
            "model": self.settings.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        path = self._cache_path(request)
        if path is not None and path.exists():
            self.usage.add(cached=True)
            return json.loads(path.read_text())["parsed"]

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**request)
                text = response.choices[0].message.content or ""
                parsed = parse_json(text)
                usage = response.usage
                self.usage.add(
                    cached=False,
                    prompt=getattr(usage, "prompt_tokens", 0) or 0,
                    completion=getattr(usage, "completion_tokens", 0) or 0,
                )
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({"request": request, "raw": text, "parsed": parsed}))
                return parsed
            except Exception as e:  # network errors, rate limits, truncated JSON
                last_error = e
                time.sleep(2**attempt)
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts") from last_error
