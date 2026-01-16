MUST follow the brief strictly. Use it as the source of truth.

You are given current requirements plus targets and missing coverage areas.
Your task: ADD ONLY the missing requirements, assumptions, or constraints to meet targets.
Do NOT rewrite or remove existing items.

Return a SINGLE JSON object (no markdown, no commentary) with this wrapper shape:
{
  "REQUIREMENTS_JSON": {"requirements":[],"assumptions":[],"constraints":[]}
}

Format contract:
- REQUIREMENTS_JSON.requirements MUST be an array of objects with {id, text, priority}.
- priority must be one of must|should|could.
- assumptions and constraints must be arrays of strings only.

Rules:
- Generate ONLY the incremental items needed to meet missing_count and missing coverage areas.
- Ensure new requirements explicitly mention missing coverage areas where needed.
- If missing_assumptions or missing_constraints is provided, add only that many items.
- No markdown, no extra keys.
