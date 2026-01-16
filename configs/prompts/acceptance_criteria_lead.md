MUST follow the brief strictly. Use it as the source of truth.

Return a SINGLE JSON object (no markdown, no commentary) with this wrapper shape:
{
  "ACCEPTANCE_CRITERIA_JSON": {"criteria":[]}
}

Format contract:
- criteria must be an array of objects with {requirement_id, criteria}.
- requirement_id must be a string (use REQ-1 style identifiers).
- criteria must be an array of strings.

Rules:
- No markdown, no extra keys.
