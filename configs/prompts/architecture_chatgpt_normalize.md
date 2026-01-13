STRICT OUTPUT CONTRACT:
Return ONLY one JSON object that matches normalized_requirements.schema.json.

Requirements:
- top-level keys MUST be: requirements, assumptions, constraints
- requirements MUST be an array of objects with ONLY: id (string), text (string), priority (must|should|could)
- assumptions MUST be an array (empty array allowed)
- constraints MUST be an array (empty array allowed)
- even if there is only one requirement, it must be inside requirements[]
- do NOT include fields not in the schema

Example (must follow schema exactly):
{"requirements":[{"id":"REQ-1","text":"System provides deterministic mock mode execution.","priority":"must"}],"assumptions":[],"constraints":[]}

No markdown, no extra text.
