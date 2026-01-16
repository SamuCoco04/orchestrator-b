MUST follow the brief strictly. Use it as the source of truth.

Input includes: brief, draft business rules, and Gemini cross-review critique.
Return a SINGLE JSON object (no markdown, no commentary):
{
  "FINAL_BUSINESS_RULES_JSON": {"rules":[]}
}

Format contract:
- FINAL_BUSINESS_RULES_JSON.rules must be an array of objects with {id, text, rationale}.

Rules:
- Apply Gemini critique to remove ambiguity, add missing detail, and cover edge cases.
- No markdown, no extra keys.
