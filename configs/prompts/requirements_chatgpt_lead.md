MUST follow the brief strictly. Use it as the source of truth.

Targets from brief:
- requirements_min={{REQ_MIN}}
- requirements_max={{REQ_MAX}}
- assumptions_min={{ASSUMPTIONS_MIN}}
- constraints_min={{CONSTRAINTS_MIN}}
- roles_expected={{ROLES_EXPECTED}}

Return a SINGLE JSON object (no markdown, no commentary) with this wrapper shape:
{
  "REVIEW_JSON": {"accepted":[],"rejected":[],"issues":[],"missing":[],"rationale":[]},
  "REQUIREMENTS_JSON": {"requirements":[],"assumptions":[],"constraints":[]}
}

Rules:
- Do NOT reuse rejected requirements to reach minimum count.
- If below minimum, generate NEW high-value requirements aligned to the brief.
- assumptions and constraints must each meet their minimums.
- requirements must meet the target range if max is provided.
- Each requirement item must have ONLY id, text, priority (must|should|could).
- No markdown, no extra keys.
