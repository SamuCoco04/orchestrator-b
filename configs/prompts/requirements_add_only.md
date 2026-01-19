MUST follow the brief strictly. Use it as the source of truth.

You are given current requirements, coverage counts, and a target count (generate_count).
Your task: ADD ONLY new requirements (exactly generate_count) without rewriting existing items.

Return a SINGLE JSON object (no markdown, no commentary) with this wrapper shape:
{
  "REQUIREMENTS_JSON": {"requirements":[],"assumptions":[],"constraints":[]}
}

Format contract:
- REQUIREMENTS_JSON.requirements MUST be an array of objects with {id, text, priority}.
- priority must be one of must|should|could.

Rules:
- Generate EXACTLY generate_count new requirements.
- Do NOT rewrite, delete, or duplicate existing requirements.
- Avoid duplicating any existing IDs or texts provided in the input.
- If missing coverage areas are listed, explicitly mention them in new requirements.
- No markdown, no extra keys.
