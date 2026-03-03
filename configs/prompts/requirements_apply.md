Use the brief as the source of truth.

Return STRICT JSON ONLY. No markdown, no prose.
Return EXACTLY one JSON object with EXACTLY these two top-level keys:
{
  "FINAL_REQUIREMENTS_JSON": {
    "requirements": [],
    "assumptions": [],
    "constraints": []
  },
  "APPLY_REPORT_JSON": {
    "applied_actions": [
      {"action": "...", "evidence": "..."}
    ],
    "unapplied_actions": []
  }
}

Rules:
- Do not output any extra top-level keys.
- FINAL_REQUIREMENTS_JSON.requirements entries must contain only: id, text, priority.
- priority must be one of: must, should, could.
- assumptions and constraints must be arrays of strings.
- applied_actions[].action should exactly match required_actions entries when present.
- applied_actions[].evidence must cite final REQ IDs.
- Output valid JSON only.
