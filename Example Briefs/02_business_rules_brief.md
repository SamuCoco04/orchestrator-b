---
artifact: business_rules
iteration_goal: "Derive precise IF/THEN rules and invariants from approved requirements without introducing new features."
quality_bar: "Every rule is traceable to one or more requirements and includes clear conditions and rationale."

token_budgets:
  lead_max_output_tokens: 1600
  apply_max_output_tokens: 1600

targets:
  target_min_items: 25
  target_max_items: 60

coverage_areas:
  - Authorization Conditions
  - Data Validation & Integrity
  - Temporal Constraints
  - State Preconditions
  - Thresholds & Limits
  - Compliance & Policy
  - Exception Handling

min_per_area: 1

inputs:
  requirements_json: "<path to requirements.json>"
  business_rules_json: "N/A"
  workflows_json: "N/A"

notes:
  - "Do NOT invent new features. Rules must be derivable from requirements."
  - "Prefer IF/THEN phrasing; include rationale for each rule."
---

## Project Context
Explain clearly what kind of system is being built.
Paste:
- Problem statement
- High-level goals
- Constraints that shape the system

## Users & Roles
List roles generically:
- Human users
- System roles
- Administrative roles
- External systems

Stress:
- Roles must be behaviorally distinct.
- Avoid role inflation.

## Scope Boundaries
Define:
- In-scope behaviors
- Out-of-scope behaviors
- Known exclusions

## Seed Requirements / Seeds
Provide:
- Requirements excerpts relevant to rules
- Regulatory obligations
- Stakeholder mandates

Clarify:
- Seeds are starting points, not final answers.

## Non-Functional Requirements (NFRs)
Provide structured placeholders:
- Security:
- Privacy:
- Performance:
- Availability:
- Scalability:
- Auditability:
- Accessibility:
- Internationalization:
- Compliance:
- Observability:

Explain how NFRs influence rules (e.g., access conditions, audit constraints).

## Inputs from Previous Artifacts
- Paste or reference the finalized requirements JSON.
- If missing, list assumptions about requirements and note gaps explicitly.

## Output Expectations
- Rules expressed as IF/THEN or invariant statements.
- Each rule includes a rationale.
- No new product features or user stories.

## Quality Checklist
- Are all rules traceable to requirements?
- Are conditions explicit (who/when/what)?
- Are invariants clearly stated?
- Are exceptions captured?

## Common Failure Modes & Fixes
- Invented features: constrain rules to requirement language.
- Vague conditions: add triggers and scope qualifiers.
- Duplicative rules: consolidate and differentiate scopes.
- Missing rationale: add a brief policy or requirement reference.
