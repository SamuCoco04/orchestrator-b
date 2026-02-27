Add ONLY assumptions and constraints to replace removed invented items and satisfy minimum counts.
Do NOT add or rewrite requirements.

Return JSON only:
{
  "ASSUMPTIONS_CONSTRAINTS_JSON": {
    "assumptions": ["..."],
    "constraints": ["..."]
  }
}

Rules:
- assumptions and constraints must be arrays of strings only.
- Keep items brief-consistent and project-agnostic.
- Do not include budget/currency/timeline claims unless explicitly in brief.
