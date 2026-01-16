---
artifact: requirements
iteration_goal: "Strengthen functional requirements coverage while eliminating ambiguity and mixed responsibilities."
quality_bar: "Each requirement is a testable shall statement with clear actor, action, and outcome. Assumptions and constraints are explicit and minimal."

token_budgets:
  lead_max_output_tokens: 2400
  apply_max_output_tokens: 2400

targets:
  target_min_items: 50
  target_max_items: 80

coverage_areas:
  - Authentication & Identity
  - Roles & Permissions
  - Core Workflows
  - Data & Storage
  - Document & Content Management
  - Notifications & Messaging
  - Integrations & External Systems
  - Administration & Configuration
  - Auditability & Logging
  - Privacy & Security
  - Accessibility
  - Internationalization
  - Observability & Monitoring
  - Error Handling & Recovery

min_per_area: 1

inputs:
  requirements_json: "N/A"
  business_rules_json: "N/A"
  workflows_json: "N/A"

notes:
  - "Keep requirements purely functional; do not include workflows or business rules."
  - "No UI or implementation decisions. Focus on behavior and outcomes."
---

## Project Context
Explain clearly what kind of system is being built.
Paste:
- Problem statement
- High-level goals
- Constraints that shape the system (budget, timeline, compliance, infrastructure)

## Users & Roles
List roles generically. Include:
- Human users
- System roles
- Administrative roles
- External systems

Stress:
- Roles must be behaviorally distinct.
- Avoid role inflation (no duplicates with minor wording changes).

## Scope Boundaries
Define:
- In-scope behaviors
- Out-of-scope behaviors
- Known exclusions

## Seed Requirements / Seeds
Provide:
- Existing requirements
- Legacy specs
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

Explain how NFRs influence functional requirements and must be reflected downstream.

## Inputs from Previous Artifacts
- None for this artifact.
- If you have legacy artifacts, paste key excerpts here or provide file paths.

## Output Expectations
- Functional requirements only (testable “shall” statements).
- Explicit assumptions and constraints as strings.
- No workflows, no business rules, no design decisions.

## Quality Checklist
- Are all requirements testable and measurable?
- Is each requirement unambiguous and complete?
- Do requirements cover all coverage areas?
- Are assumptions/constraints minimal and explicit?

## Common Failure Modes & Fixes
- Overly generic requirements: add actor, trigger, and outcome.
- Mixed responsibilities: split into atomic requirements.
- Hidden workflows: move sequencing to workflow brief.
- Unclear scope: tighten in-scope vs out-of-scope bullets.
