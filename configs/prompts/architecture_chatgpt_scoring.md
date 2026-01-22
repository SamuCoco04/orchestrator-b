STRICT OUTPUT CONTRACT:
Return ONLY valid JSON that matches scoring_result.schema.json.

Requirements:
- root object: {"scores": [...], "selected_id": "...", "rationale": "..."}
- scores[] must be a list of objects with EXACT keys: candidate_id, total_score, breakdown
- DO NOT use keys id, score, or rationale inside scores[] items
- breakdown MUST be an object with numeric values; keys must align to rubric.yaml criteria

Example (must follow schema exactly):
{"scores":[{"candidate_id":"A","total_score":0.86,"breakdown":{"feasibility":0.9,"cost":0.8,"maintainability":0.9,"scalability":0.8,"security":0.85}}],"selected_id":"A","rationale":"Best balance across rubric criteria."}

No markdown, no extra text.
