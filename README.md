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
python -m src.main --pipeline requirements --mode live --brief brief.md
```

Evidence locations for requirements runs:
- Raw prompts/responses: `runs/<run_id>/raw/*.txt`
- Parsed artifacts: `runs/<run_id>/artifacts/*.json`
- ADRs: `runs/<run_id>/artifacts/adrs/ADR-XX.md`

Requirements pipeline behavior:
- ChatGPT produces draft requirements + review
- If REQUIREMENTS_JSON is missing, the pipeline auto-retries once for the missing block
- Gemini cross-reviews
- ChatGPT applies fixes with changelog and gates
- If gates fail, a single auto-repair retry is attempted for the apply output

You can also adjust generation controls:

```bash
python -m src.main --pipeline requirements --mode live --brief brief.md --max-output-tokens 800 --temperature 0.2
```

## Pipelines

### Requirements pipeline

```bash
python -m src.main --pipeline requirements --mode live --brief brief.md
```

Outputs:
- `runs/<run_id>/artifacts/requirements.md`
- `runs/<run_id>/artifacts/requirements.json`
- `runs/<run_id>/artifacts/requirements_review.json`
- `runs/<run_id>/artifacts/changelog.json`
- `runs/<run_id>/artifacts/turnr3_gemini_cross_review.json`
- `runs/<run_id>/artifacts/adrs/ADR-XX.md`

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
