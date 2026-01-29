Add ONLY assumptions and constraints to meet the missing counts. Do NOT add or rewrite requirements.

Return a SINGLE JSON object (no markdown, no commentary) with wrapper:
{
  "REQUIREMENTS_JSON": {"requirements":[],"assumptions":[],"constraints":[]}
}

Rules:
- Generate EXACTLY missing_assumptions assumptions and EXACTLY missing_constraints constraints.
- Each assumption/constraint must be concise, specific, and testable.
- Do not repeat existing entries.
- Output ONLY JSON, no extra keys.
