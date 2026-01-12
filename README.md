# orchestrator-b

Deterministic orchestration pipeline with fixed multi-model turns, strict JSON contracts,
quality gates, and reproducible artifacts.

## Requirements

- Python 3.11+

## Quick start (mock mode)

```bash
python -m src.main --mode mock --brief path/to/brief.md
```

This creates a new run directory under `runs/<run_id>/` with raw LLM outputs, parsed JSON,
artifacts, and quality gate logs.

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
