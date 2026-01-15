You are fixing a previous APPLY output that failed quality gates.

Targets from brief:
- requirements_min={{REQ_MIN}}
- requirements_max={{REQ_MAX}}
- assumptions_min={{ASSUMPTIONS_MIN}}
- constraints_min={{CONSTRAINTS_MIN}}
- roles_expected={{ROLES_EXPECTED}}

Return a SINGLE JSON wrapper only:
{
  "FINAL_REQUIREMENTS_JSON": {"requirements":[],"assumptions":[],"constraints":[]},
  "CHANGELOG_JSON": {"splits":[],"replacements":[],"added":[],"removed":[]}
}

Rules:
- Do NOT include rejected requirements.
- If below minimum, generate NEW requirements aligned to the brief (do not resurrect rejected).
- Ensure assumptions and constraints meet minimums.
- Ensure requirements count matches the target range.
- No markdown or extra keys.
