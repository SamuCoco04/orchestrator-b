from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    raw_text: str


class LLMAdapter(Protocol):
    def complete(self, prompt: str) -> LLMResponse:
        raise NotImplementedError
