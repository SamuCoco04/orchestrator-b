from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.utils.io import write_text


def write_decision_matrix(path: Path, candidates: Dict, scoring: Dict) -> None:
    score_map = {item["candidate_id"]: item for item in scoring.get("scores", [])}
    lines: List[str] = ["candidate_id,name,total_score,rationale"]
    for candidate in candidates.get("candidates", []):
        score = score_map.get(candidate["id"], {})
        total = score.get("total_score", "")
        rationale = scoring.get("rationale", "") if candidate["id"] == scoring.get("selected_id") else ""
        lines.append(f"{candidate['id']},{candidate['name']},{total},{rationale}")
    write_text(path, "\n".join(lines) + "\n")
