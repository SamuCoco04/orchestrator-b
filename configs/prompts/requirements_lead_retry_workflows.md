MUST follow the brief strictly. Use it as the source of truth.

Return ONLY workflows. Do NOT include requirements or reviews.
Return a SINGLE JSON object (no markdown, no commentary) with this wrapper shape:
{
  "WORKFLOWS_JSON": {"workflows":[]}
}

Rules:
- Each workflow must include {id, name, states, transitions}.
- Each transition must include {from, to, trigger}.
- Use "trigger" (not "guard") and do not include extra keys.
- No markdown, no extra keys.
