MUST follow the brief strictly. Use it as the source of truth.

You are given current requirements, coverage counts, balance targets, review JSON, and a target count (generate_count).
Your task: ADD ONLY new requirements without rewriting existing items.

Return a SINGLE JSON object (no markdown, no commentary) with this wrapper shape:
{
  "REQUIREMENTS_JSON": {"requirements":[],"assumptions":[],"constraints":[]}
}

Format contract:
- REQUIREMENTS_JSON.requirements MUST be an array of objects with {id, text, priority}.
- priority must be one of must|should|could.
- assumptions and constraints must be empty arrays in add-only output.

Rules:
- Return between 1 and generate_count new requirements, and NEVER exceed generate_count.
- Do NOT rewrite, delete, or duplicate existing requirements.
- Avoid duplicating any existing IDs or texts provided in the input.
- If target_areas is provided, prioritize that distribution and map each new requirement to those areas.
- If coverage_prefix_mode is true, each new requirement MUST start with exact "[<Coverage Area>] " using only provided coverage_areas.
- If missing balance targets are listed, ensure new requirements help close those gaps.
- Each new requirement must include: actor + domain object + action + observable outcome.
- Avoid placeholders such as: "as described in the brief", "define and enforce behavior", "user-friendly interface", "provide guidelines".
- Output ONLY JSON; do NOT include assumptions or constraints in add-only output.
- Requirements must be atomic and testable.
- No markdown, no extra keys.
