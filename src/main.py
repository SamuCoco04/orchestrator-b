from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.pipeline_architecture import ArchitecturePipeline
from src.pipeline_code import CodePipeline
from src.utils.io import write_text
from src.utils.time import utc_timestamp

DEFAULT_BRIEF_TEMPLATE = """# Project Brief

## Goal
Describe the outcome you want this orchestration to deliver.

## Scope
List in-scope and out-of-scope items.

## Constraints
List technical, timing, and compliance constraints.

## Success Criteria
Define measurable success criteria.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orchestrator B")
    parser.add_argument("--mode", choices=["mock", "live"], required=True)
    parser.add_argument("--brief", required=True)
    parser.add_argument("--gate-cmd", action="append", default=[], help="Quality gate command")
    parser.add_argument("--max-output-tokens", type=int, default=800)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser


def _ensure_env(base_dir: Path) -> None:
    load_dotenv(base_dir / ".env")
    missing = [
        key
        for key in ("OPENAI_API_KEY", "GEMINI_API_KEY")
        if not os.getenv(key)
    ]
    if missing:
        missing_keys = ", ".join(missing)
        raise RuntimeError(
            "Missing required API keys: "
            f"{missing_keys}. Create a .env file from .env.example and set the keys."
        )


def main() -> None:
    args = build_parser().parse_args()
    base_dir = Path(__file__).resolve().parents[1]
    run_id = utc_timestamp()
    run_dir = base_dir / "runs" / run_id
    inputs_dir = run_dir / "inputs"
    raw_dir = run_dir / "raw"
    artifacts_dir = run_dir / "artifacts"
    quality_dir = run_dir / "quality_gates"

    for path in [inputs_dir, raw_dir, artifacts_dir, quality_dir]:
        path.mkdir(parents=True, exist_ok=True)

    os.environ["ORCH_MAX_OUTPUT_TOKENS"] = str(args.max_output_tokens)
    os.environ["ORCH_TEMPERATURE"] = str(args.temperature)

    if args.mode == "live":
        _ensure_env(base_dir)

    brief_path = Path(args.brief)
    if not brief_path.exists():
        write_text(brief_path, DEFAULT_BRIEF_TEMPLATE)
        print(
            f"Brief template created at {brief_path}. Please edit it with project details."
        )

    write_text(inputs_dir / "brief.md", brief_path.read_text(encoding="utf-8"))

    architecture = ArchitecturePipeline(args.mode, base_dir)
    architecture.run(brief_path, run_dir)

    code_pipeline = CodePipeline(args.mode, base_dir)
    code_pipeline.run(run_dir, args.gate_cmd)


if __name__ == "__main__":
    main()
