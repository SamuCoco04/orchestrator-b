MUST follow the brief strictly. Use it as the source of truth.

Input includes: brief, draft MVP scope, and Gemini cross-review critique.
Return a SINGLE JSON object (no markdown, no commentary):
{
  "FINAL_MVP_SCOPE_JSON": {"in_scope":[],"out_of_scope":[],"milestones":[]}
}

Format contract:
- in_scope and out_of_scope must be arrays of strings.
- milestones must be an array of objects with {name, description}.

Rules:
- Apply Gemini critique to remove ambiguity, add missing detail, and cover edge cases.
- No markdown, no extra keys.
