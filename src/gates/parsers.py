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


def _largest_balanced_object(text: str) -> str | None:
    candidates: list[str] = []
    for start in (idx for idx, ch in enumerate(text) if ch == "{"):
        depth = 0
        in_string = False
        escape = False
        for end in range(start, len(text)):
            ch = text[end]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == "\"":
                in_string = not in_string
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : end + 1])
                    break
    if not candidates:
        return None
    return max(candidates, key=len)


def extract_json_tolerant(raw_text: str) -> dict:
    try:
        return extract_json(raw_text)
    except ValueError:
        pass
    balanced = _largest_balanced_object(raw_text)
    if balanced:
        try:
            parsed = json.loads(balanced)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    snippet = raw_text.strip().replace("\n", " ")
    snippet = (snippet[:200] + "...") if len(snippet) > 200 else snippet
    raise ValueError(f"No JSON object found in response. Snippet: {snippet}")
