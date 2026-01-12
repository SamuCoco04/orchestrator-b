from __future__ import annotations

import os

from .llm_base import LLMAdapter, LLMResponse


class GeminiAdapter(LLMAdapter):
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

    def complete(self, prompt: str) -> LLMResponse:
        raise RuntimeError("Gemini adapter is not implemented in this mock-first repository.")
