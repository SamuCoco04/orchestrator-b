from __future__ import annotations

import json
import re
from typing import Any


def _strip_code_fences(text: str) -> str:
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return "\n".join(fenced)
    return text


def _iter_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    stripped = _strip_code_fences(text)
    candidates.append(stripped)

    for match in re.finditer(r"[\[{]", stripped):
        start = match.start()
        snippet = stripped[start:]
        candidates.append(snippet)

    return candidates


def _try_parse(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_json(raw_text: str) -> dict:
    parsed = _try_parse(raw_text)
    if parsed is not None:
        return parsed

    for candidate in _iter_json_candidates(raw_text):
        parsed = _try_parse(candidate)
        if parsed is not None:
            return parsed

    decoder = json.JSONDecoder()
    for candidate in _iter_json_candidates(raw_text):
        try:
            parsed, _ = decoder.raw_decode(candidate.lstrip())
            return parsed
        except json.JSONDecodeError:
            continue

    snippet = raw_text.strip().replace("\n", " ")
    snippet = (snippet[:200] + "...") if len(snippet) > 200 else snippet
    raise ValueError(f"No JSON object found in response. Snippet: {snippet}")
