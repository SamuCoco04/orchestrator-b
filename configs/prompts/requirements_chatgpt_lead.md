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
  "REVIEW_JSON": {"accepted":[],"rejected":[],"issues":[],"missing":[],"rationale":[]},
  "REQUIREMENTS_JSON": {"requirements":[],"assumptions":[],"constraints":[]},
  "BUSINESS_RULES_JSON": {"rules":[]},
  "WORKFLOWS_JSON": {"workflows":[]},
  "DOMAIN_MODEL_JSON": {"entities":[],"relationships":[]},
  "MVP_SCOPE_JSON": {"in_scope":[],"out_of_scope":[],"milestones":[]}
}

Format contract:
- Prefer wrapper JSON keys exactly as shown above.
- accepted/rejected MUST be arrays of requirement ID strings only (e.g., "REQ-1"), never objects.
- Each requirement item must have ONLY id, text, priority (must|should|could).
- BUSINESS_RULES_JSON.rules must be objects with {id, text, category?}.
- WORKFLOWS_JSON.workflows must include states and transitions.
- DOMAIN_MODEL_JSON must include entities and relationships.
- MVP_SCOPE_JSON must include in_scope and out_of_scope.

Rules:
- Do NOT reuse rejected requirements to reach minimum count.
- If below minimum, generate NEW high-value requirements aligned to the brief.
- assumptions and constraints must each meet their minimums.
- requirements must meet the target range if max is provided.
- If coverage_areas provided, ensure each area appears in the requirement text (include the area name).
- If seed_requirements provided, ensure each seed is captured or paraphrased in requirements.
- No markdown, no extra keys.
