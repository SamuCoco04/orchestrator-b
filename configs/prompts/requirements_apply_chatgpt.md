You are applying fixes to requirements based on review feedback.

Return plain text with TWO labeled JSON blocks exactly:
FINAL_REQUIREMENTS_JSON:
<json>
CHANGELOG_JSON:
<json>

FINAL_REQUIREMENTS_JSON schema (normalized_requirements):
{"requirements":[{"id":"REQ-1","text":"...","priority":"must"}],"assumptions":["..."],"constraints":["..."]}

CHANGELOG_JSON schema:
{"splits":[{"from":"REQ-12","to":["REQ-28","REQ-29"],"note":"..."}],"replacements":[],"added":["REQ-XX"],"removed":["REQ-YY"]}

No markdown or code fences. Output only the labeled JSON blocks.
