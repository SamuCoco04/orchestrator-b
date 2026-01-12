from __future__ import annotations

import os

from openai import OpenAI

from .llm_base import LLMAdapter


class OpenAIAdapter(LLMAdapter):
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        max_tokens = int(os.getenv("ORCH_MAX_OUTPUT_TOKENS", "800"))
        temperature = float(os.getenv("ORCH_TEMPERATURE", "0.2"))
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("OpenAI returned empty content.")
        return content
