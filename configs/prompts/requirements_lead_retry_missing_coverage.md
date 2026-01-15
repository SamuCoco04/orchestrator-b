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

You are given existing requirements and a list of missing areas/seeds.
Return ONLY additional requirements to fill the gaps. Do NOT rewrite existing requirements.

Return a SINGLE JSON object (no markdown, no commentary) with this wrapper shape:
{
  "REQUIREMENTS_JSON": {"requirements":[],"assumptions":[],"constraints":[]}
}

Rules:
- Provide ONLY new requirements; do not repeat existing ones.
- If missing_count > 0, add that many new requirements.
- If missing_areas provided, ensure each missing area appears in the requirement text (include the area name).
- If missing_seeds provided, ensure each seed is captured or paraphrased in new requirements.
- Each requirement item must have ONLY id, text, priority (must|should|could).
- No markdown, no extra keys.
