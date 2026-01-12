from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline_architecture import ArchitecturePipeline
from src.pipeline_code import CodePipeline
from src.utils.io import write_text
from src.utils.time import utc_timestamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orchestrator B")
    parser.add_argument("--mode", choices=["mock", "live"], required=True)
    parser.add_argument("--brief", required=True)
    parser.add_argument("--gate-cmd", action="append", default=[], help="Quality gate command")
    return parser


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

    brief_path = Path(args.brief)
    write_text(inputs_dir / "brief.md", brief_path.read_text(encoding="utf-8"))

    architecture = ArchitecturePipeline(args.mode, base_dir)
    architecture.run(brief_path, run_dir)

    code_pipeline = CodePipeline(args.mode, base_dir)
    code_pipeline.run(run_dir, args.gate_cmd)


if __name__ == "__main__":
    main()
