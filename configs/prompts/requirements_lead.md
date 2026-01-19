MUST follow the brief strictly. Use it as the source of truth.

Targets from brief:
- requirements_min={{REQ_MIN}}
- requirements_max={{REQ_MAX}}
- assumptions_min={{ASSUMPTIONS_MIN}}
- constraints_min={{CONSTRAINTS_MIN}}
- roles_expected={{ROLES_EXPECTED}}
- coverage_areas={{COVERAGE_AREAS}}
- min_per_area={{MIN_PER_AREA}}
- seed_requirements:
{{SEED_REQUIREMENTS}}

Return a SINGLE JSON object (no markdown, no commentary) with this wrapper shape:
{
  "REQUIREMENTS_JSON": {"requirements":[],"assumptions":[],"constraints":[]}
}

Format contract:
- REQUIREMENTS_JSON.requirements MUST be an array of objects with {id, text, priority}.
- priority must be one of must|should|could.
- assumptions and constraints must be arrays of strings only.

Rules:
- Do NOT reuse rejected requirements (if any).
- If below minimum, generate NEW high-value requirements aligned to the brief.
- requirements must meet the target range if max is provided.
- If coverage_areas provided, ensure each area appears in requirement text (include the area name).
- If seed_requirements provided, ensure each seed is captured or paraphrased in requirements.
- No markdown, no extra keys.
