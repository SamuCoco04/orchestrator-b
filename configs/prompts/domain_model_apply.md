MUST follow the brief strictly. Use it as the source of truth.

Input includes: brief, draft domain model, and Gemini cross-review critique.
Return a SINGLE JSON object (no markdown, no commentary):
{
  "FINAL_DOMAIN_MODEL_JSON": {"entities":[],"relationships":[]}
}

Format contract:
- entities must be an array of objects with {name, description, attributes}.
- attributes must be an array of objects with {name, type, description}.
- relationships must be an array of objects with {from, to, type, description}.

Rules:
- Apply Gemini critique to remove ambiguity, add missing detail, and cover edge cases.
- No markdown, no extra keys.
