Fix prefix format for requirements when coverage_prefix_mode is true.

Input includes invalid_requirements (id,text,priority) and coverage_areas.
Return JSON only:
{
  "PREFIX_FIXED_JSON": {
    "requirements": [
      {"id":"REQ-001","text":"[<Coverage Area>] ...","priority":"must"}
    ]
  }
}

Rules:
- Keep semantic meaning of each requirement text; only add/correct prefix.
- Prefix must be exactly one allowed coverage area string: "[<Coverage Area>] ".
- Keep id and priority unchanged.
