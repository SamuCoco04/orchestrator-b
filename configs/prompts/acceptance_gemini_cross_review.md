Review the acceptance criteria quality. Focus on missing edge cases, ambiguity, and non-verifiable criteria.
Return a SINGLE JSON object (no markdown, no commentary) with this shape:
{
  "issues": [],
  "missing": [],
  "rationale": []
}

Rules:
- List concrete issues with requirement_id references when possible.
- Flag criteria that are not testable.
- Suggest missing cases succinctly.
