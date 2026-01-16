MUST follow the brief strictly. Use it as the source of truth.

Input includes: brief, draft workflows, and Gemini cross-review critique.
Return a SINGLE JSON object (no markdown, no commentary):
{
  "FINAL_WORKFLOWS_JSON": {"workflows":[]}
}

Format contract:
- FINAL_WORKFLOWS_JSON.workflows must be an array of objects with {id, name, states, transitions}.
- states must be an array of strings.
- transitions must be an array of objects with {from, to, trigger}.

Rules:
- Apply Gemini critique to remove ambiguity, add missing detail, and cover edge cases.
- No markdown, no extra keys.
