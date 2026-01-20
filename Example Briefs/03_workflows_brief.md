---
artifact: workflows
iteration_goal: "Model core workflows as explicit state machines tied to requirements and business rules."
quality_bar: "Each workflow has mutually exclusive states, clear triggers, and traces to requirements/rules."

token_budgets:
  lead_max_output_tokens: 2000
  apply_max_output_tokens: 2000

targets:
  target_min_items: 5
  target_max_items: 12

coverage_areas:
  - Primary User Journeys
  - Administrative Overrides
  - Error & Recovery Paths
  - External System Interactions
  - Approval or Review Flows

min_per_area: 1

inputs:
  requirements_json: "<path to requirements.json>"
  business_rules_json: "<path to business_rules.json>"
  workflows_json: "N/A"

notes:
  - "Every workflow must map to requirements or business rules."
  - "Use triggers, not guards; states must be mutually exclusive."
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
- Core requirement excerpts driving workflows
- Related business rules

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

Explain how NFRs influence workflow states and transitions.

## Inputs from Previous Artifacts
- Paste or reference requirements and business rules.
- If missing, list assumptions and mark workflows as provisional.

## Output Expectations
- Workflows with states and transitions.
- Triggers must be explicit events.
- No business rules embedded; reference them as triggers if needed.

## Quality Checklist
- Are states mutually exclusive and collectively meaningful?
- Are triggers explicit and deterministic?
- Are error/recovery paths modeled?
- Does each workflow map to requirements or rules?

## Common Failure Modes & Fixes
- Guards instead of triggers: rewrite as events.
- Hidden workflows inside rules: move to workflow brief.
- Overlapping states: split or rename for exclusivity.
- Missing failure paths: add explicit error states.
