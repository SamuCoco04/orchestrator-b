MUST follow the brief strictly. Use it as the source of truth.

Return a SINGLE JSON wrapper (no markdown, no commentary):
{
  "FINAL_DOMAIN_MODEL_JSON": {"entities":[],"relationships":[]},
  "FINAL_MVP_SCOPE_JSON": {"in_scope":[],"out_of_scope":[],"milestones":[]}
}

Rules:
- FINAL_DOMAIN_MODEL_JSON.entities must include {name, description, attributes}.
- FINAL_DOMAIN_MODEL_JSON.relationships must include {from, to, type, description}.
- FINAL_MVP_SCOPE_JSON.in_scope/out_of_scope must be arrays of strings.
- No markdown, no extra keys.
