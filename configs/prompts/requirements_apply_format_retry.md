You must re-emit the SAME content from the previous apply output using the exact wrapper JSON schema.

Return a SINGLE JSON object (no markdown, no commentary) with this wrapper shape:
{
  "FINAL_REQUIREMENTS_JSON": {"requirements":[],"assumptions":[],"constraints":[]}
}

Rules:
- Re-emit the same requirements, assumptions, and constraints content.
- Do NOT add new requirements or edit meaning unless required to fit the schema.
- Output ONLY JSON, no extra keys.

Input includes: brief and the previous apply_raw text.
