from __future__ import annotations

import os
from google import genai

from .llm_base import LLMAdapter


class GeminiAdapter(LLMAdapter):
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        self.client = genai.Client(api_key=self.api_key)

        # Use a stable alias by default (recommended naming convention: gemini-flash-latest)
        # Allow overriding via .env
        self.model = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
        except Exception as e:
            # Give a helpful message for the common "model not found" issue.
            msg = (
                f"Gemini generate_content failed for model='{self.model}'.\n"
                f"Tip: set GEMINI_MODEL in your .env to an available model alias.\n"
                f"Common working alias: gemini-flash-latest\n"
                f"Original error: {e}"
            )
            raise RuntimeError(msg) from e

        if not response or not getattr(response, "text", None):
            raise RuntimeError("Gemini returned empty content.")

        return response.text
