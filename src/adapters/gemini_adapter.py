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

    def generate(self, prompt: str) -> str:
        max_tokens = int(os.getenv("ORCH_MAX_OUTPUT_TOKENS", "800"))
        temperature = float(os.getenv("ORCH_TEMPERATURE", "0.2"))
        response = self.client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
                "response_mime_type": "application/json",
            },
        )
        if not response.text:
            raise RuntimeError("Gemini returned empty content.")
        return response.text
