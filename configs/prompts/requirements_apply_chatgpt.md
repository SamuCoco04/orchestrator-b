MUST follow the brief strictly. Use it as the source of truth.

Input includes: brief, draft requirements, review, and Gemini cross-review.
Return a SINGLE JSON wrapper (no markdown, no commentary):
{
  "FINAL_REQUIREMENTS_JSON": {"requirements":[],"assumptions":[],"constraints":[]},
  "CHANGELOG_JSON": {"splits":[],"replacements":[],"added":[],"removed":[]}
}

Rules:
- FINAL_REQUIREMENTS_JSON must include ONLY accepted requirements plus NEW requirements added to meet goals.
- If any draft requirement is rejected, it must NOT appear in final.
- If under minimum (30–45), generate NEW high-value requirements aligned to the brief (do not resurrect rejected).
- assumptions and constraints must each have at least 3 items.
- CHANGELOG_JSON must list added/removed/replacements/splits (arrays may be minimal but must exist).
- No markdown, no extra keys.
