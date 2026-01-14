You are fixing a previous APPLY output that failed quality gates.

Return a SINGLE JSON wrapper only:
{
  "FINAL_REQUIREMENTS_JSON": {"requirements":[],"assumptions":[],"constraints":[]},
  "CHANGELOG_JSON": {"splits":[],"replacements":[],"added":[],"removed":[]}
}

Rules:
- Do NOT include rejected requirements.
- If below minimum, generate NEW requirements aligned to the brief (do not resurrect rejected).
- Ensure assumptions and constraints have at least 3 items each.
- Ensure requirements count is 30–45.
- No markdown or extra keys.
