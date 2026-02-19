from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from src.utils.io import write_text


@dataclass
class GateResult:
    command: str
    return_code: int
    output: str


def run_commands(commands: Iterable[str], log_dir: Path) -> List[GateResult]:
    results: List[GateResult] = []
    log_dir.mkdir(parents=True, exist_ok=True)
    for index, command in enumerate(commands, start=1):
        completed = subprocess.run(
            command,
            shell=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output = completed.stdout or ""
        log_path = log_dir / f"gate_{index}.log"
        write_text(log_path, output)
        results.append(GateResult(command=command, return_code=completed.returncode, output=output))
    return results


def is_done(test_results: Iterable[GateResult], audit_report: dict) -> bool:
    tests_ok = all(result.return_code == 0 for result in test_results)
    has_p0 = any(item.get("severity") == "P0" for item in audit_report.get("findings", []))
    return tests_ok and not has_p0
