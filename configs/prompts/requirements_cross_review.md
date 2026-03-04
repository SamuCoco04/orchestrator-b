OUTPUT STRICT JSON ONLY. NO MARKDOWN. NO PROSE.

Project-agnostic critic review: use ONLY provided brief fields and requirements payload.

Return EXACTLY one JSON object with this shape and these keys only:
{
  "blocking_issues": [],
  "required_actions": [],
  "weak_requirements": [],
  "missing_areas": []
}

Rules:
- blocking_issues: must-fix defects that prevent acceptance.
- required_actions: imperative, testable actions to fix defects.
- weak_requirements: requirement IDs (REQ-xxx) that are vague or non-testable.
- missing_areas: concrete missing coverage areas derived from brief fields.
- If there are no issues, return empty arrays.
