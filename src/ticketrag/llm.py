"""OpenAI wrapper: schema-constrained JSON output, on-disk response cache, token accounting.

The cache makes reruns (and evals) free and deterministic; usage is tracked so cost is visible."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODEL = "gpt-4o-mini"


class LLM:
    def __init__(self, model: str | None = None, cache_dir: str | Path = "data/cache/llm"):
        from openai import OpenAI

        load_dotenv()
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        self.client = OpenAI()  # reads OPENAI_API_KEY from the environment
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def _key(self, *parts) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()

    def json(self, system: str, user: str, schema: dict, name: str = "result",
             temperature: float = 0.0, max_tokens: int = 1000, retries: int = 3) -> dict:
        key = self._key(self.model, system, user, schema, temperature)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            self.usage["cache_hits"] += 1
            return json.loads(cache_file.read_text(encoding="utf-8"))

        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": name, "schema": schema, "strict": True}},
            max_completion_tokens=max_tokens,
        )
        if not self.model.startswith(("gpt-5", "o1", "o3", "o4")):  # reasoning models fix temperature
            kwargs["temperature"] = temperature

        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                self.usage["calls"] += 1
                self.usage["prompt_tokens"] += resp.usage.prompt_tokens
                self.usage["completion_tokens"] += resp.usage.completion_tokens
                choice = resp.choices[0]
                if choice.finish_reason != "stop":  # truncated / filtered output is not valid JSON
                    raise ValueError(f"finish_reason={choice.finish_reason}")
                out = json.loads(choice.message.content)
                break
            except Exception:  # rate limit, transient network error, truncated or invalid JSON
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        cache_file.write_text(json.dumps(out), encoding="utf-8")
        return out
