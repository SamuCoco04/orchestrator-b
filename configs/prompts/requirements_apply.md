Use the brief as the source of truth.

Return STRICT JSON ONLY. No markdown, no prose.
Return a SINGLE JSON object with EXACTLY these top-level keys:
{
  "FINAL_REQUIREMENTS_JSON": {
    "requirements": [],
    "assumptions": [],
    "constraints": []
  },
  "APPLY_REPORT_JSON": {
    "applied_actions": [],
    "unapplied_actions": []
  }
}

Rules:
- FINAL_REQUIREMENTS_JSON.requirements entries must contain only: id, text, priority.
- priority must be one of: must, should, could.
- assumptions and constraints must be arrays of strings.
- APPLY_REPORT_JSON.applied_actions entries must include action and evidence fields.
- APPLY_REPORT_JSON.unapplied_actions must be an array (empty if all applied).
- applied_actions[].action must exactly match required_actions strings when provided.
- applied_actions[].evidence must cite final REQ IDs.
- Output valid JSON only.
