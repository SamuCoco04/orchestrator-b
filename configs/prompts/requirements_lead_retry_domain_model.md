MUST follow the brief strictly. Use it as the source of truth.

Return ONLY the domain model. Do NOT include requirements or reviews.
Return a SINGLE JSON object (no markdown, no commentary) with this wrapper shape:
{
  "DOMAIN_MODEL_JSON": {"entities":[],"relationships":[]}
}

Rules:
- Each entity must include {name, description, attributes}.
- Each attribute must include {name, type, description}.
- Each relationship must include {from, to, type, description}.
- No markdown, no extra keys.
