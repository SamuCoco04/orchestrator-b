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
  "FINAL_BUSINESS_RULES_JSON": {"rules":[]},
  "FINAL_WORKFLOWS_JSON": {"workflows":[]},
  "FINAL_DOMAIN_MODEL_JSON": {"entities":[],"relationships":[]},
  "FINAL_MVP_SCOPE_JSON": {"in_scope":[],"out_of_scope":[],"milestones":[]},
  "CHANGELOG_JSON": {"splits":[],"replacements":[],"added":[],"removed":[]}
}

Format contract:
- Prefer wrapper JSON keys exactly as shown above.
- FINAL_REQUIREMENTS_JSON items must have ONLY id, text, priority (must|should|could).
- CHANGELOG_JSON.splits must be an array of objects with keys {from, into}.
- CHANGELOG_JSON.added/replacements/removed must be arrays of requirement ID strings.
- FINAL_REQUIREMENTS_JSON.requirements MUST be an array of objects only (no strings, no mixed items).
- FINAL_BUSINESS_RULES_JSON.rules must be objects with {id, text, rationale}.
- FINAL_WORKFLOWS_JSON.workflows must include states and transitions.
- FINAL_DOMAIN_MODEL_JSON must include entities and relationships.
- FINAL_MVP_SCOPE_JSON must include in_scope and out_of_scope.
- FINAL_MVP_SCOPE_JSON.milestones must be an array of objects with {name, description}.

Example requirements array:
[
  {"id":"REQ-1","text":"The system must log all access attempts.","priority":"must"},
  {"id":"REQ-2","text":"The system should export reports in CSV format.","priority":"should"}
]

Rules:
- FINAL_REQUIREMENTS_JSON must include ONLY accepted requirements plus NEW requirements added to meet goals.
- If any draft requirement is rejected, it must NOT appear in final.
- If under minimum, generate NEW requirements aligned to the brief (do not resurrect rejected).
- assumptions and constraints must meet their minimums.
- CHANGELOG_JSON must list added/removed/replacements/splits (arrays may be minimal but must exist).
- No markdown, no extra keys.
