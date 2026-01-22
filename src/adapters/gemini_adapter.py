from __future__ import annotations

import os
import random
import time
from typing import List

from google import genai

from .llm_base import LLMAdapter


class GeminiAdapter(LLMAdapter):
    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        self.client = genai.Client(api_key=api_key)

        primary = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
        self.model_candidates: List[str] = [
            primary,
            "gemini-pro",
            "gemini-1.5-pro",
        ]

        self.max_attempts = int(os.getenv("GEMINI_MAX_ATTEMPTS", "5"))
        self.base_delay = float(os.getenv("GEMINI_BASE_DELAY_SECONDS", "1.0"))

    def _is_transient(self, err: Exception) -> bool:
        msg = str(err).lower()
        return any(s in msg for s in ["503", "unavailable", "429", "too many", "timeout", "temporarily"])

    def generate(self, prompt: str) -> str:
        last_err: Exception | None = None

        for model in self.model_candidates:
            for attempt in range(1, self.max_attempts + 1):
                try:
                    print(f"[gemini] model={model} attempt={attempt}/{self.max_attempts}")
                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                    )
                    text = getattr(response, "text", None)
                    if not text:
                        raise RuntimeError("Gemini returned empty content.")
                    return text

                except Exception as e:
                    last_err = e
                    if not self._is_transient(e):
                        break

                    delay = self.base_delay * (2 ** (attempt - 1)) + random.random() * 0.5
                    print(f"[gemini] transient error: {e} -> sleeping {delay:.2f}s")
                    time.sleep(delay)

            print(f"[gemini] switching model after failures: {model}")

        raise RuntimeError(
            "Gemini generate_content failed for all candidate models. "
            f"Last error: {last_err}"
        ) from last_err
