from __future__ import annotations

import json


def extract_json(raw_text: str) -> dict:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in response.")
    payload = raw_text[start : end + 1]
    return json.loads(payload)
