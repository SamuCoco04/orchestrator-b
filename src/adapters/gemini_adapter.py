from __future__ import annotations

import os
import random
import time
from typing import Dict, List

from google import genai

from .llm_base import LLMAdapter, LLMResponse


class GeminiUnavailableError(RuntimeError):
    pass


class GeminiAdapter(LLMAdapter):
    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        self.client = genai.Client(api_key=api_key)

        models_env = os.getenv("ORCH_GEMINI_MODELS", "")
        if models_env.strip():
            self.model_candidates = [item.strip() for item in models_env.split(",") if item.strip()]
        else:
            self.model_candidates = [
                "gemini-2.5-pro",
                "gemini-2.5-flash",
            ]

        self.max_attempts = int(os.getenv("GEMINI_MAX_ATTEMPTS", "5"))
        self.base_delay = float(os.getenv("GEMINI_BASE_DELAY_SECONDS", "1.0"))
        self._available_models_cache: set[str] | None = None
        self._selected_model: str | None = None
        self._unusable_models: set[str] = set()
        self._fallbacks: List[Dict[str, str]] = []

    def _available_models(self) -> set[str]:
        if self._available_models_cache is not None:
            return self._available_models_cache
        available: set[str] = set()
        listed = self.client.models.list()
        for entry in listed:
            name = getattr(entry, "name", None)
            if isinstance(name, str) and name.strip():
                available.add(name.strip())
                available.add(name.strip().split("/")[-1])
        self._available_models_cache = available
        return available

    def _model_in_available(self, model: str, available: set[str]) -> bool:
        return model in available or f"models/{model}" in available

    def _pick_model(self) -> str:
        available = self._available_models()
        for model in self.model_candidates:
            if model in self._unusable_models:
                continue
            if self._model_in_available(model, available):
                self._selected_model = model
                return model
        available_sorted = sorted(available)
        raise GeminiUnavailableError(
            "No preferred Gemini model available. "
            f"Preferred={self.model_candidates}. "
            f"Available(sample)={available_sorted[:20]}"
        )

    def _record_fallback(self, model: str, err: Exception) -> None:
        self._fallbacks.append({"model": model, "error_code_or_message": str(err)})

    def _is_not_found_or_unsupported(self, err: Exception) -> bool:
        msg = str(err).lower()
        tokens = ["404", "not found", "unsupported", "generatecontent", "not supported"]
        return any(token in msg for token in tokens)

    def _call_generate_content(self, model: str, prompt: str, max_tokens: int | None):
        generation_config = {"max_output_tokens": max_tokens} if max_tokens is not None else None
        if generation_config is None:
            return self.client.models.generate_content(model=model, contents=prompt)
        try:
            return self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=generation_config,
            )
        except TypeError:
            return self.client.models.generate_content(
                model=model,
                contents=prompt,
                generation_config=generation_config,
            )

    def get_diagnostics(self) -> Dict[str, object]:
        available = sorted(self._available_models_cache or [])
        return {
            "selected_model": self._selected_model,
            "available_models_sample": available[:20],
            "fallbacks": list(self._fallbacks),
        }

    def _is_transient(self, err: Exception) -> bool:
        msg = str(err).lower()
        return any(s in msg for s in ["503", "unavailable", "429", "too many", "timeout", "temporarily"])

    def complete(self, prompt: str, max_tokens: int | None = None) -> LLMResponse:
        response_text = self.generate(prompt, max_tokens=max_tokens)
        return LLMResponse(raw_text=response_text)

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        last_err: Exception | None = None

        while True:
            try:
                model = self._pick_model()
            except GeminiUnavailableError:
                if last_err is not None:
                    raise GeminiUnavailableError(
                        "Gemini models unavailable after fallbacks. "
                        f"Last error: {last_err}"
                    ) from last_err
                raise
            for attempt in range(1, self.max_attempts + 1):
                try:
                    print(f"[gemini] model={model} attempt={attempt}/{self.max_attempts}")
                    response = self._call_generate_content(model, prompt, max_tokens)
                    text = getattr(response, "text", None)
                    if not text:
                        raise RuntimeError("Gemini returned empty content.")
                    return text

                except Exception as e:
                    last_err = e
                    if self._is_not_found_or_unsupported(e):
                        self._record_fallback(model, e)
                        self._unusable_models.add(model)
                        break
                    if not self._is_transient(e):
                        self._record_fallback(model, e)
                        break

                    delay = self.base_delay * (2 ** (attempt - 1)) + random.random() * 0.5
                    print(f"[gemini] transient error: {e} -> sleeping {delay:.2f}s")
                    time.sleep(delay)

            print(f"[gemini] switching model after failures: {model}")
            self._unusable_models.add(model)
            if len(self._unusable_models) >= len(self.model_candidates):
                break

        raise GeminiUnavailableError(
            "Gemini generate_content failed for all candidate models. "
            f"Last error: {last_err}"
        ) from last_err
