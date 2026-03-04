Return STRICT JSON ONLY. No markdown, no prose.

You are repairing malformed apply output.
Return exactly one JSON object with this shape:
{
  "FINAL_REQUIREMENTS_JSON": {
    "requirements": [],
    "assumptions": [],
    "constraints": []
  }
}

Rules:
- Preserve requirement meaning from input.
- Ensure requirements is an array of requirement objects.
- Ensure assumptions has at least 3 items.
- Ensure constraints has at least 3 items.
- Output valid JSON only.
