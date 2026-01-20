Return ONLY valid JSON matching this schema. No markdown, no extra text.

Targets from brief:
- requirements_min={{REQ_MIN}}
- requirements_max={{REQ_MAX}}
- assumptions_min={{ASSUMPTIONS_MIN}}
- constraints_min={{CONSTRAINTS_MIN}}
- coverage_areas={{COVERAGE_AREAS}}
- min_per_area={{MIN_PER_AREA}}

You must:
- Compute per-area counts using the draft requirements and coverage_areas (case-insensitive).
- Identify missing coverage areas and list them in missing_areas.
- Identify overlapping/duplicate topics and list them in duplication_suspects.
- Identify generic requirements and list them in too_generic with reason only.
- Identify domain-specific gaps and list them in domain_missing.
- required_actions MUST include explicit tasks for every gap found above.

Return JSON with this exact shape:
{
  "missing_areas": ["..."],
  "duplication_suspects": ["..."],
  "too_generic": [{"id":"REQ-001","reason":"..."}],
  "domain_missing": ["..."],
  "required_actions": [
    {"action":"add_requirements","count":0,"areas":["..."]},
    {"action":"strengthen_specificity","ids":["REQ-001"]},
    {"action":"dedupe_requirements","ids":["REQ-001","REQ-002"]}
  ]
}
