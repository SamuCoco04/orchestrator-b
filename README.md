# orchestrator-b

Deterministic orchestration pipeline with fixed multi-model turns, strict JSON contracts,
quality gates, and reproducible artifacts.

## Requirements

- Python 3.11+

## Quick start (mock mode)

```bash
python -m src.main --pipeline requirements --mode mock --brief path/to/brief.md
```

Windows setup helper:

```powershell
scripts\setup.ps1
```

This creates a new run directory under `runs/<run_id>/` with raw LLM outputs, parsed JSON,
artifacts, and quality gate logs.

## Live mode setup

Create a `.env` file in the repository root (you can copy `.env.example`) and add:

- `OPENAI_API_KEY`: Create one at https://platform.openai.com/account/api-keys
- `GEMINI_API_KEY`: Create one at https://aistudio.google.com/app/apikey

Then run the requirements pipeline:

```bash
python -m src.main --pipeline requirements --mode live --brief brief.md --artifact requirements
```

Evidence locations for requirements runs:
- Raw prompts/responses: `runs/<run_id>/raw/*.txt`
- Parsed artifacts: `runs/<run_id>/artifacts/*.json`
- ADRs: `runs/<run_id>/artifacts/adrs/ADR-XX.md`

Requirements pipeline behavior (single artifact mode):
- ChatGPT produces a draft artifact
- Gemini cross-reviews that artifact for gaps/ambiguity
- ChatGPT applies fixes and outputs a FINAL_* JSON wrapper
- If validation fails, a targeted retry fixes missing/invalid fields only

You can also adjust generation controls:

```bash
python -m src.main --pipeline requirements --mode live --brief brief.md --max-output-tokens 800 --temperature 0.2
```


## Brief frontmatter targets

You can set per-project targets by adding YAML frontmatter to the brief:

```yaml
---
requirements_target:
  min: 30
  max: 60
assumptions_min: 3
constraints_min: 3
roles_expected: ["Student", "Administrator", "Coordinator"]
artifact_token_budgets:
  requirements: 2400
  workflows: 2000
---
```

When omitted, defaults are used. These targets drive quality gates and prompt rendering.

## Pipelines

### Requirements pipeline

```bash
python -m src.main --pipeline requirements --mode live --brief brief.md --artifact requirements
```

Run the four core artifact phases sequentially:

```bash
python -m src.main --pipeline requirements --mode live --brief brief.md --artifact requirements
python -m src.main --pipeline requirements --mode live --brief brief.md --artifact business_rules
python -m src.main --pipeline requirements --mode live --brief brief.md --artifact workflows
python -m src.main --pipeline requirements --mode live --brief brief.md --artifact domain_model
```

Artifacts land in `runs/<run_id>/artifacts/` as `{artifact}.json` and `{artifact}.md` plus
raw evidence under `runs/<run_id>/raw/` for each draft/cross-review/apply step.

### Architecture pipeline

```bash
python -m src.main --pipeline architecture --mode live --brief brief.md
```

Outputs:
- `runs/<run_id>/artifacts/decision_matrix.csv`
- `runs/<run_id>/artifacts/architecture_summary.md`
- `runs/<run_id>/artifacts/adrs/ADR-XX.md`

### Code pipeline

```bash
python -m src.main --pipeline code --mode live --brief brief.md --gate-cmd "pytest -q"
```

Load requirements/architecture from a previous run:

```bash
python -m src.main --pipeline code --mode live --brief brief.md --inputs-from-run <run_id>
```

Outputs:
- `runs/<run_id>/artifacts/turnc1_code_tasks.json`
- `runs/<run_id>/artifacts/audit_report.json`
- `runs/<run_id>/artifacts/audit_report_chatgpt.json`
- `runs/<run_id>/quality_gates/*.log`

## Real provider setup

Set environment variables (see `.env.example`):

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`

Then pass `--mode live` to enable real adapters. The adapters read keys from the environment.

## Option C readiness (state + scheduler)

This design keeps each pipeline stage deterministic, idempotent, and artifacts-first. Every
turn is stored as raw + parsed JSON, and the run directory acts as a complete state snapshot.
A scheduler can later treat each stage as a resumable job, keyed by the run directory, and
move between stages based on the presence/validity of artifacts.
