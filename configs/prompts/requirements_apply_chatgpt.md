MUST follow the brief strictly. Use it as the source of truth.

Targets from brief:
- requirements_min={{REQ_MIN}}
- requirements_max={{REQ_MAX}}
- assumptions_min={{ASSUMPTIONS_MIN}}
- constraints_min={{CONSTRAINTS_MIN}}
- roles_expected={{ROLES_EXPECTED}}

Input includes: brief, draft requirements, review, and Gemini cross-review.
Return a SINGLE JSON wrapper (no markdown, no commentary):
{
  "FINAL_REQUIREMENTS_JSON": {"requirements":[],"assumptions":[],"constraints":[]},
  "CHANGELOG_JSON": {"splits":[],"replacements":[],"added":[],"removed":[]}
}

Rules:
- FINAL_REQUIREMENTS_JSON must include ONLY accepted requirements plus NEW requirements added to meet goals.
- If any draft requirement is rejected, it must NOT appear in final.
- If under minimum, generate NEW requirements aligned to the brief (do not resurrect rejected).
- assumptions and constraints must meet their minimums.
- CHANGELOG_JSON must list added/removed/replacements/splits (arrays may be minimal but must exist).
- No markdown, no extra keys.
