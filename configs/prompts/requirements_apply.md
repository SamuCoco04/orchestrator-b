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
You MUST implement every required_actions item.

Return a SINGLE JSON object (no markdown, no commentary) with ONLY these top-level keys:
{
  "FINAL_REQUIREMENTS_JSON": {"requirements":[],"assumptions":[],"constraints":[]},
  "APPLY_REPORT_JSON": {
    "applied_actions": [
      {
        "action":"<exact string copied from required_actions>",
        "evidence":"<short evidence mentioning REQ-IDs>"
      }
    ],
    "unapplied_actions": ["<exact required_action>"],
    "fixed_weak_requirements": ["REQ-001"],
    "removed_invented_constraints": ["<short quote>"]
  }
}

Rules:
- FINAL_REQUIREMENTS_JSON must remain schema-valid and contain ONLY requirements/assumptions/constraints.
- APPLY_REPORT_JSON must always exist (arrays may be empty).
- applied_actions[].action MUST exactly match a required_actions string.
- applied_actions[].evidence MUST cite existing final REQ-IDs.
- fixed_weak_requirements must include IDs you rewrote/split.
- removed_invented_constraints should list brief-inconsistent assumptions/constraints removed.
- If coverage_prefix_mode is true, each requirement MUST start with "[<Coverage Area>] " using only provided coverage_areas.
- Do NOT invent budget/cost/timeline constraints unless explicitly present in brief text.
- Never output a single requirement object.
- If output is long, MINIFY JSON.
