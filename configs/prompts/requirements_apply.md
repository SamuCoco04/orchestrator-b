MUST follow the brief strictly. Use it as the source of truth.

Targets from brief:
- requirements_min={{REQ_MIN}}
- requirements_max={{REQ_MAX}}
- assumptions_min={{ASSUMPTIONS_MIN}}
- constraints_min={{CONSTRAINTS_MIN}}
- roles_expected={{ROLES_EXPECTED}}

Input includes: brief, draft requirements, and Gemini cross-review critique.
Return a SINGLE JSON wrapper (no markdown, no commentary):
{
  "FINAL_REQUIREMENTS_JSON": {"requirements":[],"assumptions":[],"constraints":[]}
}

Format contract:
- FINAL_REQUIREMENTS_JSON.requirements MUST be an array of objects with {id, text, priority}.
- priority must be one of must|should|could.
- assumptions and constraints must be arrays of strings only.

Rules:
- Apply Gemini critique to remove ambiguity, add missing detail, and cover edge cases.
- If below minimum, generate NEW requirements aligned to the brief.
- No markdown, no extra keys.
