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
You MUST implement every item in required_actions. Ensure requirements are atomic and testable.

Return a SINGLE JSON object (no markdown, no commentary) with ONLY these top-level keys:
{
  "FINAL_REQUIREMENTS_JSON": {"requirements":[],"assumptions":[],"constraints":[]},
  "APPLY_REPORT_JSON": {
    "applied_actions": [{"action":"<exact string from required_actions>","evidence":"<short>"}],
    "unapplied_actions": ["<exact action string>"]
  }
}

Format contract:
- FINAL_REQUIREMENTS_JSON.requirements MUST be an array of objects with {id, text, priority}.
- priority must be one of must|should|could.
- assumptions and constraints must be arrays of strings only.
- APPLY_REPORT_JSON.applied_actions must include ALL required_actions with short evidence.
- APPLY_REPORT_JSON.unapplied_actions must be empty when all required_actions are satisfied.
- APPLY_REPORT_JSON must always be present (even if arrays are empty).
- Each applied_actions[].evidence MUST mention at least one requirement ID that exists in FINAL_REQUIREMENTS_JSON.requirements[].id.
- Output ONE JSON object only (no markdown).
- Must include top-level key FINAL_REQUIREMENTS_JSON with keys: requirements, assumptions, constraints.
- Never output a single requirement object.
- If output is long, MINIFY JSON (no extra whitespace) to stay within token limits.

Rules:
- Apply Gemini critique to remove ambiguity, add missing detail, and cover edge cases.
- If below minimum, generate NEW requirements aligned to the brief.
- Do NOT rename existing requirement IDs; only add new requirements with new IDs.
- No markdown, no extra keys.

Example APPLY_REPORT_JSON:
{
  "applied_actions": [
    {"action":"Add missing coverage for notifications","evidence":"Added REQ-042 and REQ-043 to cover notification delivery and failure handling."}
  ],
  "unapplied_actions": []
}
