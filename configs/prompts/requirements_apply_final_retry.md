You are fixing issues reported by a final Gemini review of the FINAL requirements.

Rules:
- Address ONLY the issues listed in final_review.
- Do NOT rewrite unrelated requirements.
- Output a SINGLE JSON object (no markdown, no commentary).

Return JSON with this shape:
{
  "FINAL_REQUIREMENTS_JSON": {"requirements":[],"assumptions":[],"constraints":[]},
  "APPLY_REPORT_JSON": {"applied_actions":[],"unresolved_actions":[]}
}
