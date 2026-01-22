---
artifact: mvp_scope
iteration_goal: "Define a realistic MVP scope with explicit exclusions and staged milestones."
quality_bar: "In-scope and out-of-scope lists are explicit, and milestones are concrete, testable, and time-ordered."

token_budgets:
  lead_max_output_tokens: 1200
  apply_max_output_tokens: 1200

targets:
  target_min_items: 8
  target_max_items: 20

coverage_areas:
  - Core Capabilities
  - Operational Readiness
  - Compliance Minimums
  - Integration Minimums
  - Support & Administration

min_per_area: 1

inputs:
  requirements_json: "<path to requirements.json>"
  business_rules_json: "<path to business_rules.json or N/A>"
  workflows_json: "N/A"

notes:
  - "Milestones must be objects with {name, description}."
  - "Focus on scope tradeoffs, not implementation tasks."
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
- Requirements that must be in MVP
- Requirements that can be deferred

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

Explain how NFRs influence MVP scope and exclusions.

## Inputs from Previous Artifacts
- Paste or reference requirements (and optionally business rules).
- If missing, document assumptions and defer uncertain items.

## Output Expectations
- In-scope list: explicit, capability-level items.
- Out-of-scope list: clear exclusions.
- Milestones: list of {name, description} objects.

## Quality Checklist
- Are scope tradeoffs explicit and justified?
- Are exclusions concrete and non-overlapping?
- Do milestones describe outcomes, not tasks?
- Can stakeholders agree on scope without guessing?

## Common Failure Modes & Fixes
- Vague scope: rewrite as concrete capability outcomes.
- Hidden features: move to out-of-scope or future milestone.
- Milestones as tasks: reframe to deliverable outcomes.
- Missing exclusions: add explicit deferrals.
