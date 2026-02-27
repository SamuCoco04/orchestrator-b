MUST follow the brief strictly. Use it as the source of truth.

Return a SINGLE JSON object (no markdown, no commentary) with this wrapper shape:
{
  "WORKFLOWS_JSON": {"workflows":[]}
}

Format contract:
- WORKFLOWS_JSON.workflows must be an array of objects with {id, name, states, transitions}.
- states must be an array of strings.
- transitions must be an array of objects with {from, to, trigger}.

Rules:
- Keep workflows aligned to the brief.
- No markdown, no extra keys.
