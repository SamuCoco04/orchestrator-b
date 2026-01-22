MUST follow the brief strictly. Use it as the source of truth.

Input includes: brief, draft acceptance criteria, and Gemini cross-review critique.
Return a SINGLE JSON object (no markdown, no commentary):
{
  "FINAL_ACCEPTANCE_CRITERIA_JSON": {"criteria":[]}
}

Format contract:
- criteria must be an array of objects with {requirement_id, criteria}.
- requirement_id must be a string (use REQ-1 style identifiers).
- criteria must be an array of strings.

Rules:
- Apply Gemini critique to remove ambiguity, add missing detail, and cover edge cases.
- No markdown, no extra keys.
