MUST follow the brief strictly. Use it as the source of truth.

Return ONLY business rules. Do NOT include requirements or reviews.
Return a SINGLE JSON object (no markdown, no commentary) with this wrapper shape:
{
  "BUSINESS_RULES_JSON": {"rules":[]}
}

Rules:
- Each rule must include {id, text, rationale}.
- Keep rules concise and aligned to the brief.
- No markdown, no extra keys.
