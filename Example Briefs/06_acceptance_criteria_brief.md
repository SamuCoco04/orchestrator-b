---
artifact: acceptance_criteria
iteration_goal: "Translate requirements into verifiable acceptance criteria with edge cases and negative scenarios."
quality_bar: "Every criterion is testable, uses Given/When/Then, and covers boundary conditions."

token_budgets:
  lead_max_output_tokens: 2000
  apply_max_output_tokens: 2000

targets:
  target_min_items: 30
  target_max_items: 80

coverage_areas:
  - Happy Paths
  - Negative Scenarios
  - Boundary Conditions
  - Role-Based Behavior
  - Error Handling
  - Compliance Checks

min_per_area: 1

inputs:
  requirements_json: "<path to requirements.json>"
  business_rules_json: "<path to business_rules.json>"
  workflows_json: "<path to workflows.json>"

notes:
  - "Use Given/When/Then format for each criterion."
  - "Include negative scenarios and boundary cases."
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
- Requirements to verify
- Regulatory or contractual obligations
- Critical edge cases from stakeholders

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

Explain how NFRs influence acceptance criteria and test cases.

## Inputs from Previous Artifacts
- Paste or reference requirements, business rules, and workflows.
- If missing, state assumptions and flag risks in criteria.

## Output Expectations
- Acceptance criteria per requirement.
- Given/When/Then structure.
- Include negative scenarios and boundary cases.

## Quality Checklist
- Is each criterion testable and unambiguous?
- Do criteria map to requirements and rules?
- Are negative and boundary cases included?
- Can QA execute without extra interpretation?

## Common Failure Modes & Fixes
- Vague criteria: add measurable outcomes.
- Missing negative tests: add explicit failure cases.
- Poor traceability: reference requirement IDs.
- Overly broad scenarios: split into atomic cases.
