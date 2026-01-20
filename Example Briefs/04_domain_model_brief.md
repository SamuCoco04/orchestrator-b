---
artifact: domain_model
iteration_goal: "Define entities, attributes, and relationships that enable the behaviors in requirements and workflows."
quality_bar: "Every entity and relationship is justified by system behavior, and every relationship has a clear description."

token_budgets:
  lead_max_output_tokens: 1600
  apply_max_output_tokens: 1600

targets:
  target_min_items: 10
  target_max_items: 25

coverage_areas:
  - Core Entities
  - Supporting Entities
  - Audit & Logging Entities
  - External Integration Entities
  - Configuration Entities

min_per_area: 1

inputs:
  requirements_json: "<path to requirements.json>"
  business_rules_json: "<path to business_rules.json>"
  workflows_json: "<path to workflows.json>"

notes:
  - "Every relationship MUST have a description."
  - "Entities must be justified by behavior, not implementation convenience."
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
- Requirements excerpts that imply domain entities
- Business rules that indicate entities or relationships
- Workflow states that imply entity lifecycle

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

Explain how NFRs influence data attributes and relationships.

## Inputs from Previous Artifacts
- Paste or reference requirements, business rules, and workflows.
- If missing, state assumptions and call out gaps.

## Output Expectations
- Entities with attributes (name/type/description).
- Relationships with from/to/type/description.
- No implementation details (tables, services, APIs).

## Quality Checklist
- Does every entity map to behavior or rules?
- Are attributes meaningful and not redundant?
- Are relationships described and justified?
- Can an architect infer data boundaries from this model?

## Common Failure Modes & Fixes
- Orphan entities: justify via requirements or remove.
- Missing relationship descriptions: add the behavior link.
- Overly technical fields: rewrite as business attributes.
- Duplicate entities: merge and clarify scope.
