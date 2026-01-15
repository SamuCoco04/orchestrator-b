MUST follow the brief strictly. Use it as the source of truth.

You will receive MUST requirements only.
Return a SINGLE JSON object (no markdown, no commentary) with this wrapper shape:
{
  "ACCEPTANCE_CRITERIA_JSON": {
    "criteria": [
      {"requirement_id": "", "criteria": []}
    ]
  }
}

Rules:
- Provide 3-6 clear, testable criteria per MUST requirement.
- Each criterion must be objectively verifiable.
- requirement_id must match the MUST requirements provided.
- No markdown, no extra keys.
