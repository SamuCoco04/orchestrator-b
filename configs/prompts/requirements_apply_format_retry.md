Wrap the existing content into FINAL_REQUIREMENTS_JSON with requirements:[...], assumptions:[...], constraints:[...].

Return a SINGLE JSON object (no markdown, no commentary) with this wrapper shape:
{ "FINAL_REQUIREMENTS_JSON": {"requirements":[],"assumptions":[],"constraints":[]} }

Rules:
- Output MUST be a SINGLE JSON object (no markdown fences, no commentary).
- JSON must be strictly valid (double quotes, no trailing commas).
- Re-emit the same requirements, assumptions, and constraints content.
- Do NOT add new requirements or edit meaning unless required to fit the schema.
- Output ONLY JSON, no extra keys.

Input includes: brief and the previous apply_raw text.
