Return ONLY valid JSON matching this schema. No markdown, no extra text.

Targets from brief:
- requirements_min={{REQ_MIN}}
- requirements_max={{REQ_MAX}}
- assumptions_min={{ASSUMPTIONS_MIN}}
- constraints_min={{CONSTRAINTS_MIN}}
- coverage_areas={{COVERAGE_AREAS}}
- min_per_area={{MIN_PER_AREA}}

You must:
- Compute per-area counts using the draft requirements and coverage_areas (case-insensitive unless coverage_prefix_mode is true).
- Identify missing coverage areas and list them in missing_areas.
- Identify blocking issues as plain strings in blocking_issues.
- Identify weak requirements (umbrella or non-testable) in weak_requirements by ID.
- required_actions MUST include explicit action strings for every gap found above.

Return JSON with this exact shape:
{
  "blocking_issues": ["..."],
  "missing_areas": ["..."],
  "weak_requirements": ["REQ-001"],
  "required_actions": ["..."]
}
