# orchestrator-b

Deterministic orchestration pipeline with fixed multi-model turns, strict JSON contracts,
quality gates, and reproducible artifacts.

## Requirements

- Python 3.11+

## Quick start (mock mode)

```bash
python -m src.main --mode mock --brief path/to/brief.md
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

Then run:

```bash
python -m src.main --mode live --brief brief.md
```

You can also adjust generation controls:

```bash
python -m src.main --mode live --brief brief.md --max-output-tokens 800 --temperature 0.2
```

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
