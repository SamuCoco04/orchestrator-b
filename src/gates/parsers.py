from __future__ import annotations

import json
import re
from typing import Any


def _strip_code_fences(text: str) -> str:
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return "\n".join(fenced)
    return text


def _clean_text(text: str) -> str:
    cleaned = text.lstrip("\ufeff")
    cleaned = cleaned.strip()
    cleaned = _strip_code_fences(cleaned)
    cleaned = re.sub(r"^[\x00-\x1f\x7f]+", "", cleaned)
    cleaned = re.sub(r"[\x00-\x1f\x7f]+$", "", cleaned)
    return cleaned


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
    cleaned = _clean_text(raw_text)
    parsed = _try_parse(cleaned)
    if parsed is not None:
        return parsed

    for candidate in _iter_json_candidates(cleaned):
        parsed = _try_parse(candidate)
        if parsed is not None:
            return parsed

    decoder = json.JSONDecoder()
    for candidate in _iter_json_candidates(cleaned):
        try:
            parsed, _ = decoder.raw_decode(candidate.lstrip())
            return parsed
        except json.JSONDecodeError:
            continue

    snippet = cleaned.strip().replace("\n", " ")
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
    cleaned = _clean_text(raw_text)
    try:
        return extract_json(cleaned)
    except ValueError:
        pass
    balanced_candidates: list[str] = []
    balanced = _largest_balanced_object(cleaned)
    if balanced:
        balanced_candidates.append(balanced)
    for candidate in _iter_json_candidates(cleaned):
        balanced = _largest_balanced_object(candidate)
        if balanced:
            balanced_candidates.append(balanced)
    best: dict | None = None
    best_len = 0
    for candidate in balanced_candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and len(candidate) > best_len:
            best = parsed
            best_len = len(candidate)
    if best is not None:
        return best
    snippet = cleaned.strip().replace("\n", " ")
    snippet = (snippet[:200] + "...") if len(snippet) > 200 else snippet
    raise ValueError(f"No JSON object found in response. Snippet: {snippet}")
