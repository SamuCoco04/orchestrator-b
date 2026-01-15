Return ONLY the FINAL_REQUIREMENTS_JSON block.

Targets from brief:
- requirements_min={{REQ_MIN}}
- requirements_max={{REQ_MAX}}
- assumptions_min={{ASSUMPTIONS_MIN}}
- constraints_min={{CONSTRAINTS_MIN}}
- roles_expected={{ROLES_EXPECTED}}

Format:
FINAL_REQUIREMENTS_JSON:
{"requirements":[],"assumptions":[],"constraints":[]}

Rules:
- requirements must be objects only (no strings or mixed items).
- assumptions and constraints must meet their minimums.
- No markdown, no extra text.
