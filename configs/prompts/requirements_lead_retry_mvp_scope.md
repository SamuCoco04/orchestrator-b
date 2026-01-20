MUST follow the brief strictly. Use it as the source of truth.

Return ONLY MVP scope. Do NOT include requirements or reviews.
Return a SINGLE JSON object (no markdown, no commentary) with this wrapper shape:
{
  "MVP_SCOPE_JSON": {"in_scope":[],"out_of_scope":[],"milestones":[]}
}

Rules:
- in_scope and out_of_scope must be arrays of strings.
- milestones must be an array of objects with {name, description}.
- No markdown, no extra keys.
