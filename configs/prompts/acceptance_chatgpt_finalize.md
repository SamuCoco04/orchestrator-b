You will receive MUST requirements, draft acceptance criteria, and a cross review.
Return a SINGLE JSON object (no markdown, no commentary) with this wrapper shape:
{
  "ACCEPTANCE_CRITERIA_JSON": {
    "criteria": [
      {"requirement_id": "", "criteria": []}
    ]
  }
}

Rules:
- Improve clarity and testability based on the cross review.
- Keep requirement_id values aligned to the MUST requirements provided.
- No markdown, no extra keys.
