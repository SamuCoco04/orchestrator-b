Return ONLY valid JSON matching the cross-review schema. No markdown, no extra text.

Targets from brief:
- requirements_min={{REQ_MIN}}
- requirements_max={{REQ_MAX}}
- assumptions_min={{ASSUMPTIONS_MIN}}
- constraints_min={{CONSTRAINTS_MIN}}
- coverage_areas={{COVERAGE_AREAS}}
- min_per_area={{MIN_PER_AREA}}

You must:
- Compute per-area counts using the draft requirements and coverage_areas (case-insensitive).
- If min_per_area is provided, set target=min_per_area for each area.
- For each area, populate missing_coverage with current, target, and add counts.
- Identify generic requirements and list them in too_generic with reason and fix.
- Identify overlaps or duplicates and list in duplicates_or_overlaps.
- required_actions MUST include explicit tasks for every gap found above.

Return JSON with this exact shape:
{
  "review_version": "1.0",
  "missing_coverage": [{"area":"","current":0,"target":0,"add":0}],
  "too_generic": [{"id":"","reason":"","fix":""}],
  "duplicates_or_overlaps": [{"ids":["",""],"reason":"","merge_or_split":""}],
  "required_actions": [
    {"type":"ADD","area":null,"count":0,"ids":[],"instruction":""}
  ],
  "notes": ["..."]
}
