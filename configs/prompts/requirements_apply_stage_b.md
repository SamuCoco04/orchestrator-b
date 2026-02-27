MUST follow the brief strictly. Use it as the source of truth.

Return a SINGLE JSON wrapper (no markdown, no commentary):
{
  "FINAL_BUSINESS_RULES_JSON": {"rules":[]},
  "FINAL_WORKFLOWS_JSON": {"workflows":[]}
}

Rules:
- FINAL_BUSINESS_RULES_JSON.rules must be objects with {id, text, rationale}.
- FINAL_WORKFLOWS_JSON.workflows must include {id, name, states, transitions}.
- Each transition must include {from, to, trigger}.
- Use "trigger" (not "guard") and do not include extra keys.
- No markdown, no extra keys.
