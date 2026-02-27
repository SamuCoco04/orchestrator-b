You are fixing ENFORCEMENT GAPS only.

Return ONE JSON object with EXACT keys:
{
  "FINAL_REQUIREMENTS_JSON": {"requirements":[],"assumptions":[],"constraints":[]},
  "APPLY_REPORT_JSON": {
    "applied_actions": [{"action":"<exact missing action>","evidence":"<REQ-IDs>"}],
    "unapplied_actions": [],
    "fixed_weak_requirements": ["REQ-001"],
    "removed_invented_constraints": ["<short quote>"]
  }
}

Rules:
- Input includes current FINAL_REQUIREMENTS_JSON and missing_actions only.
- Apply only what is needed to satisfy missing_actions.
- action strings must exactly match missing_actions entries.
- evidence must mention final REQ-IDs.
- Output JSON only, no prose.
