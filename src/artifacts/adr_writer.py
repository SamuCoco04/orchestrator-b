from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.utils.io import write_text


def write_adr(path: Path, adr: Dict) -> None:
    lines: List[str] = [
        f"# {adr['adr_id']}: {adr['title']}",
        "",
        f"Status: {adr['status']}",
        "",
        "## Context",
        adr["context"],
        "",
        "## Decision",
        adr["decision"],
        "",
        "## Consequences",
    ]
    lines.extend([f"- {item}" for item in adr.get("consequences", [])])
    write_text(path, "\n".join(lines) + "\n")
