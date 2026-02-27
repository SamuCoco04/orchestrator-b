Fix the JSON formatting only. Preserve meaning; do not add or remove requirements.

Return a SINGLE JSON object only, no markdown, no commentary:
{
  "REQUIREMENTS_JSON": {"requirements":[],"assumptions":[],"constraints":[]}
}

Rules:
- Output MUST be strict JSON.
- Output MUST be minified (single line; no pretty formatting or newlines).
- Do NOT include unescaped double quotes inside any "text" strings.
- If quoting is needed, use single quotes or parentheses.
- Requirements IDs must remain strings in REQ-001 format.
- Arrays MUST separate elements with commas; never emit adjacent objects without commas.
