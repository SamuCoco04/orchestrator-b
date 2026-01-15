from __future__ import annotations

import os
import time

from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, InternalServerError

from .llm_base import LLMAdapter, LLMResponse


class OpenAIAdapter(LLMAdapter):
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=self.api_key)

    def complete(self, prompt: str) -> LLMResponse:
        max_tokens = int(os.getenv("ORCH_MAX_OUTPUT_TOKENS", "800"))
        temperature = float(os.getenv("ORCH_TEMPERATURE", "0.2"))
        attempt = 0
        backoff = 1.0
        while True:
            attempt += 1
            try:
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
                usage = getattr(response, "usage", None)
                if usage:
                    usage_payload = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "total_tokens": getattr(usage, "total_tokens", None),
                    }
                    print(
                        "[openai] model=gpt-4o-mini "
                        f"prompt_tokens={usage_payload['prompt_tokens']} "
                        f"completion_tokens={usage_payload['completion_tokens']} "
                        f"total_tokens={usage_payload['total_tokens']}"
                    )
                else:
                    usage_payload = None
                    print("[openai] usage not provided by SDK")
                result = LLMResponse(raw_text=content)
                setattr(result, "usage", usage_payload)
                return result
            except RateLimitError as exc:
                error = getattr(exc, "error", None)
                code = getattr(error, "code", None)
                if code == "insufficient_quota":
                    raise RuntimeError(
                        "OpenAI API quota exceeded. Please enable billing in your OpenAI account."
                    ) from exc
                if attempt >= 4:
                    raise
            except (APITimeoutError, APIConnectionError, InternalServerError):
                if attempt >= 4:
                    raise
            time.sleep(backoff)
            backoff *= 2

    def generate(self, prompt: str) -> str:
        return self.complete(prompt).raw_text
