OUTPUT STRICT JSON ONLY. NO MARKDOWN. NO PROSE.

Targets from brief:
- requirements_min={{REQ_MIN}}
- requirements_max={{REQ_MAX}}
- assumptions_min={{ASSUMPTIONS_MIN}}
- constraints_min={{CONSTRAINTS_MIN}}
- coverage_areas={{COVERAGE_AREAS}}
- min_per_area={{MIN_PER_AREA}}

You must:
- Critique only against the brief and seed_sources. Stay project-agnostic.
- Identify must-fix issues in blocking_issues.
- List required_actions as concise imperative strings that are directly checkable.
- weak_requirements must contain IDs only (REQ-xxx) for vague/umbrella/non-testable requirements.
- missing_domain_topics must list concrete brief-derived domain gaps.
- invented_constraints_flags must list assumptions/constraints that look invented (budget/timeline/etc.) and not grounded in the brief.
- If coverage_prefix_mode is true, count strict prefix compliance "[<Coverage Area>] " and report unmapped_count.

Return this exact JSON shape:
{
  "blocking_issues": ["..."],
  "required_actions": ["..."],
  "weak_requirements": ["REQ-001"],
  "missing_domain_topics": ["..."],
  "invented_constraints_flags": ["..."],
  "coverage_findings": {
    "unmapped_count": 0,
    "missing_areas": ["..."]
  }
}

It must output a single JSON object with exactly these keys: blocking_issues, required_actions, weak_requirements, missing_domain_topics, invented_constraints_flags, coverage_findings.
