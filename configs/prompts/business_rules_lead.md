MUST follow the brief strictly. Use it as the source of truth.

Return a SINGLE JSON object (no markdown, no commentary) with this wrapper shape:
{
  "BUSINESS_RULES_JSON": {"rules":[]}
}

Format contract:
- BUSINESS_RULES_JSON.rules must be an array of objects with {id, text, rationale}.

Rules:
- Keep rules concise and directly grounded in the brief.
- No markdown, no extra keys.
