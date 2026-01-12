Return ONLY valid JSON that matches normalized_requirements.schema.json.

Requirements for JSON:
- root object: {"requirements": [...], "assumptions": [...], "constraints": [...]}
- requirements[] must be a list of atomic requirement statements.
- each requirement item MUST have ONLY these fields: id (string), text (string), priority (must|should|could).
- convert any role/responsibility content into individual requirement text entries with id and priority.
- do NOT include fields not in the schema (e.g., role, responsibilities, description).

Example (must follow schema exactly):
{"requirements":[{"id":"REQ-1","text":"System provides deterministic mock mode execution.","priority":"must"}],"assumptions":["No external API keys in mock mode."],"constraints":["Outputs must be valid JSON."]}

No markdown, no extra text.
