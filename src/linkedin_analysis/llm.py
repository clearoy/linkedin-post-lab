"""Small OpenAI-compatible chat client with retries, usage counts, and a disk cache.

Two providers, configured in `.env` at the project root (see `.env.example`):
  deepseek    DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_BASE_URL     (feature discovery)
  openrouter  OPENROUTER_API_KEY, JEV_MODEL, OPENROUTER_BASE_URL      (Jev judge)

`LLMClient` calls chat completions (DeepSeek). `DecisionsClient` calls
OpenRouter's decisions endpoint, the only way to reach Jev: it takes one
`state` and many typed questions per request and returns probabilities.
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

CACHE_DIR = PROCESSED / "llm_cache"

PROVIDERS = {
    # provider: (key var, model var, base url var, default model, default base url)
    "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "DEEPSEEK_BASE_URL", "deepseek-v4-pro", "https://api.deepseek.com"),
    "openrouter": ("OPENROUTER_API_KEY", "JEV_MODEL", "OPENROUTER_BASE_URL", "typesafe/jev-1.13", "https://openrouter.ai/api/v1"),
}


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    api_key: str
    model: str
    base_url: str

    @classmethod
    def from_env(cls, provider: str, model: str | None = None, env_file: Path = ROOT / ".env") -> "LLMSettings":
        key_var, model_var, url_var, default_model, default_url = PROVIDERS[provider]
        load_dotenv(env_file)
        key = os.getenv(key_var, "").strip()
        if not key:
            raise RuntimeError(f"{key_var} is empty. Put your key in {env_file}.")
        return cls(
            provider=provider,
            api_key=key,
            model=model or os.getenv(model_var) or default_model,
            base_url=os.getenv(url_var) or default_url,
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
    cost_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, cached: bool, usage: dict | None = None) -> None:
        usage = usage or {}
        with self._lock:
            self.calls += 1
            self.cached += cached
            if not cached:
                # chat completions use prompt/completion, decisions use input/output
                self.prompt_tokens += usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                self.completion_tokens += usage.get("completion_tokens") or usage.get("output_tokens") or 0
                self.cost_usd += usage.get("cost") or 0.0  # OpenRouter reports cost

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class TruncatedResponse(RuntimeError):
    """The reply hit max_tokens. Retrying the same request will not help."""


class LLMClient:
    """Chat completions with a request-hash disk cache.

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

    def complete(self, request: dict, validate=None) -> dict:
        """Send a raw chat.completions request; return the response as a plain dict.

        `validate(response_dict)` may raise to trigger a retry (e.g. bad JSON).
        Only responses that pass validation are cached.
        """
        request = {"model": self.settings.model, **request}
        path = self._cache_path(request)
        if path is not None and path.exists():
            self.usage.add(cached=True)
            return json.loads(path.read_text())["response"]

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**request).model_dump()
                choice = response["choices"][0]
                if choice.get("finish_reason") == "length" and validate is not None:
                    raise TruncatedResponse(
                        f"reply hit max_tokens={request.get('max_tokens')} "
                        f"(usage {response.get('usage')}); reasoning counts toward it, so raise max_tokens"
                    )
                if validate is not None:
                    validate(response)
                self.usage.add(cached=False, usage=response.get("usage"))
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({"request": request, "response": response}))
                return response
            except TruncatedResponse:
                raise
            except Exception as e:  # network errors, rate limits, invalid output
                last_error = e
                time.sleep(2**attempt)
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_error}") from last_error

    def chat_json(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 32000) -> dict:
        """Chat call that returns the reply parsed as a JSON object."""
        request = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        response = self.complete(request, validate=lambda r: parse_json(r["choices"][0]["message"]["content"] or ""))
        return parse_json(response["choices"][0]["message"]["content"])


class DecisionsClient:
    """OpenRouter decisions endpoint (POST /api/alpha/decisions), with cache and retries.

    Uses one pooled HTTP connection set, so many threads can share it.
    Retries rate limits (429), server errors (5xx) and network errors with
    exponential backoff; other 4xx errors fail immediately.
    """

    RETRY_STATUS = {408, 429, 500, 502, 503, 524, 529}

    def __init__(self, settings: LLMSettings, cache_dir: Path | None = CACHE_DIR, max_retries: int = 6,
                 timeout: float = 120.0, max_connections: int = 64):
        import httpx

        self.settings = settings
        self.url = settings.base_url.split("/api/")[0] + "/api/alpha/decisions"
        self.http = httpx.Client(
            headers={"Authorization": f"Bearer {settings.api_key}", "Content-Type": "application/json"},
            timeout=timeout,
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        )
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.usage = Usage()

    def _cache_path(self, body: dict) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(json.dumps({"endpoint": "decisions", **body}, sort_keys=True).encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def decide(self, state, questions: dict, validate=None) -> dict:
        """Send one decisions request and return the response JSON."""
        import random

        import httpx

        body = {"model": self.settings.model, "state": state, "questions": questions}
        path = self._cache_path(body)
        if path is not None and path.exists():
            self.usage.add(cached=True)
            return json.loads(path.read_text())["response"]

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = self.http.post(self.url, json=body)
                if r.status_code != 200:
                    error = RuntimeError(f"decisions HTTP {r.status_code}: {r.text[:500]}")
                    if r.status_code not in self.RETRY_STATUS:
                        raise error
                    last_error = error
                else:
                    response = r.json()
                    if validate is not None:
                        validate(response)
                    self.usage.add(cached=False, usage=response.get("usage"))
                    if path is not None:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps({"request": body, "response": response}))
                    return response
            except (httpx.TransportError, ValueError, KeyError) as e:  # network, bad JSON, failed validation
                last_error = e
            time.sleep(min(2**attempt, 30) + random.random())
        raise RuntimeError(f"decisions call failed after {self.max_retries} attempts: {last_error}") from last_error
