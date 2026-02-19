MUST follow the brief strictly. Use it as the source of truth.

You are given a list of generic requirements that must be expanded into atomic, testable requirements.
Return replacements that map each generic requirement ID to 3-8 atomic requirements.
Do NOT keep the original generic requirement.

Return a SINGLE JSON object (no markdown, no commentary) with this wrapper shape:
{
  "REPLACEMENTS_JSON": {
    "replacements": [
      {"from": "REQ-001", "into": [{"id":"REQ-001A","text":"...","priority":"must"}]}
    ]
  }
}

Format contract:
- Each replacement entry must include from and into.
- into must be an array of objects with {id, text, priority}.
- priority must be one of must|should|could.

Rules:
- Each replacement must be atomic and testable.
- No markdown, no extra keys.
