You are fixing unresolved blocking actions from the Gemini review.

Rules:
- Address ONLY the unresolved blocking actions listed in the input errors.
- Do NOT rewrite unrelated requirements.
- Output a SINGLE JSON object (no markdown, no commentary).

Return JSON with this shape:
{
  "FINAL_REQUIREMENTS_JSON": {"requirements":[],"assumptions":[],"constraints":[]},
  "CHANGELOG_JSON": {"splits":[],"replacements":[],"added":[],"removed":[]},
  "APPLY_REPORT_JSON": {"applied_actions":[],"unresolved_actions":[]}
}
