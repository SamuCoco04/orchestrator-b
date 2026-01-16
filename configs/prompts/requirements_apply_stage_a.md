MUST follow the brief strictly. Use it as the source of truth.

Return a SINGLE JSON wrapper (no markdown, no commentary):
{
  "FINAL_REQUIREMENTS_JSON": {"requirements":[],"assumptions":[],"constraints":[]},
  "CHANGELOG_JSON": {"splits":[],"replacements":[],"added":[],"removed":[]}
}

Rules:
- FINAL_REQUIREMENTS_JSON items must have ONLY id, text, priority (must|should|could).
- FINAL_REQUIREMENTS_JSON.requirements MUST be an array of objects only.
- assumptions and constraints must be arrays of strings only.
- CHANGELOG_JSON.splits must be an array of objects with keys {from, into}.
- CHANGELOG_JSON.added/replacements/removed must be arrays of requirement ID strings.
- No markdown, no extra keys.
