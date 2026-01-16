MUST follow the brief strictly. Use it as the source of truth.

Return a SINGLE JSON object (no markdown, no commentary) with this wrapper shape:
{
  "DOMAIN_MODEL_JSON": {"entities":[],"relationships":[]}
}

Format contract:
- entities must be an array of objects with {name, description, attributes}.
- attributes must be an array of objects with {name, type, description}.
- relationships must be an array of objects with {from, to, type, description}.

Rules:
- Keep entities and relationships grounded in the brief.
- No markdown, no extra keys.
