MUST follow the brief strictly. Use it as the source of truth.

Output a SINGLE JSON object (no markdown, no commentary) with this wrapper shape: { "REVIEW_JSON": {"accepted":[],"rejected":[],"issues":[],"missing":[],"rationale":[]}, "REQUIREMENTS_JSON": {"requirements":[],"assumptions":[],"constraints":[]} }

Rules:

REQUIRE Erasmus Coordinator role requirements (deadlines, approvals, validation, publishing).
Do NOT reuse rejected requirements to reach minimum count.
If below minimum, generate NEW high-value requirements aligned to the brief.
assumptions and constraints MUST each have at least 3 items.
requirements MUST contain 30–45 items with unique IDs.
Each requirement item must have ONLY id, text, priority (must|should|could).
No markdown, no extra keys.