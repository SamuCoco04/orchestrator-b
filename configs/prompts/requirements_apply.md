MUST follow the brief strictly. Use it as the source of truth.

Targets from brief:
- requirements_min={{REQ_MIN}}
- requirements_max={{REQ_MAX}}
- assumptions_min={{ASSUMPTIONS_MIN}}
- constraints_min={{CONSTRAINTS_MIN}}
- roles_expected={{ROLES_EXPECTED}}
- coverage_areas={{COVERAGE_AREAS}}
- min_per_area={{MIN_PER_AREA}}

Input includes: brief, draft requirements, and Gemini cross-review JSON.
You MUST implement every item in required_actions. If required_actions includes ADD with a count, you MUST add exactly that many requirements.
Ensure requirements are atomic and testable.

Return a SINGLE JSON wrapper (no markdown, no commentary):
{
  "FINAL_REQUIREMENTS_JSON": {"requirements":[],"assumptions":[],"constraints":[]},
  "CHANGELOG_JSON": {"splits":[],"replacements":[],"added":[],"removed":[]}
}

Format contract:
- FINAL_REQUIREMENTS_JSON.requirements MUST be an array of objects with {id, text, priority}.
- priority must be one of must|should|could.
- assumptions and constraints must be arrays of strings only.
- CHANGELOG_JSON.splits must be objects with {from, into}.
- CHANGELOG_JSON.added/replacements/removed must be arrays of requirement ID strings.

Rules:
- Apply Gemini critique to remove ambiguity, add missing detail, and cover edge cases.
- If below minimum, generate NEW requirements aligned to the brief.
- No markdown, no extra keys.
