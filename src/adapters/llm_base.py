from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    raw_text: str


class LLMAdapter(Protocol):
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def complete(self, prompt: str) -> LLMResponse:
        return LLMResponse(raw_text=self.generate(prompt))
