Return ONLY valid JSON matching normalized_requirements.schema.json.

Requirements:
- top-level keys: requirements, assumptions, constraints
- requirements is an array of objects with ONLY: id, text, priority (must|should|could)
- assumptions and constraints are arrays (empty allowed)

Example:
{"requirements":[{"id":"REQ-1","text":"Add Erasmus Coordinator role with deadline tracking.","priority":"must"}],"assumptions":[],"constraints":[]}

No markdown, no extra text.
