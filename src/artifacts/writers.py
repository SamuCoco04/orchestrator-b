from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.utils.io import write_text


def write_requirements(path: Path, normalized: Dict) -> None:
    lines: List[str] = ["# Requirements", ""]
    for req in normalized.get("requirements", []):
        lines.append(f"- **{req['id']}** ({req['priority']}): {req['text']}")
    if normalized.get("assumptions"):
        lines.extend(["", "## Assumptions"])
        lines.extend([f"- {item}" for item in normalized["assumptions"]])
    if normalized.get("constraints"):
        lines.extend(["", "## Constraints"])
        lines.extend([f"- {item}" for item in normalized["constraints"]])
    write_text(path, "\n".join(lines) + "\n")
