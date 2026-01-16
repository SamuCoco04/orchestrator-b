# Example Briefs

## What these briefs are
These are engineering-oriented templates to drive AI-assisted requirements analysis.
They are designed for iterative, artifact-by-artifact generation and emphasize depth,
traceability, and schema compliance. Each brief targets a single artifact so outputs can
support architecture design rather than shallow documentation.

## Recommended execution order
1. Requirements
2. Business Rules
3. Workflows
4. Domain Model
5. MVP Scope
6. Acceptance Criteria

## How to use them
1. Copy the brief to your own working folder.
2. Fill in project-specific content.
3. Run the pipeline for ONE artifact.
4. Inspect output quality against the checklist.
5. Iterate on the brief before moving to the next artifact.

## Example CLI commands (PowerShell-friendly)
```powershell
python -m src.main --pipeline requirements --artifact requirements --brief ./briefs/requirements.md
python -m src.main --pipeline requirements --artifact business_rules --brief ./briefs/business_rules.md
python -m src.main --pipeline requirements --artifact workflows --brief ./briefs/workflows.md
python -m src.main --pipeline requirements --artifact domain_model --brief ./briefs/domain_model.md
python -m src.main --pipeline requirements --artifact mvp_scope --brief ./briefs/mvp_scope.md
python -m src.main --pipeline requirements --artifact acceptance_criteria --brief ./briefs/acceptance_criteria.md
```

## Important principle
Project-specific constraints MUST live in briefs. Prompt templates must remain generic.
