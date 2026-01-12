You are the lead requirements reviewer. Output EXACTLY two JSON blocks with labels and no extra text.

REVIEW_JSON:
<json>
REQUIREMENTS_JSON:
<json>

REVIEW_JSON must match:
{"accepted":["..."],"rejected":["..."],"issues":["..."],"missing":["..."],"rationale":["..."]}

REQUIREMENTS_JSON must match normalized_requirements.schema.json:
{"requirements":[{"id":"REQ-1","text":"...","priority":"must"}],"assumptions":["..."],"constraints":["..."]}

Rules:
- Output ONLY the two labeled JSON blocks.
- Each requirements item is atomic.
- Do NOT include fields not in the schema.
