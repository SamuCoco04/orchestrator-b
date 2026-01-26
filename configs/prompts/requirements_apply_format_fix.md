You must re-emit the SAME content from the previous apply output using the exact wrapper JSON schema.

Output MUST be a SINGLE JSON object only (no markdown, no commentary).
JSON must be strictly valid: double quotes only, no trailing commas, minified (no extra whitespace).

Return ONLY this wrapper shape:
{ "FINAL_REQUIREMENTS_JSON": { "requirements": [], "assumptions": [], "constraints": [] } }

Rules:
- Re-emit the same requirements, assumptions, and constraints content.
- Do NOT add new requirements or edit meaning unless required to fit the schema.
- Keep JSON minified to fit within token limits.

Input includes: brief and the previous apply_raw text.
