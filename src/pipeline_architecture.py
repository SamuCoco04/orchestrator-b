from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from jsonschema import validate

from src.adapters.gemini_adapter import GeminiAdapter
from src.adapters.llm_base import LLMAdapter
from src.adapters.mock_adapter import MockAdapter
from src.adapters.openai_adapter import OpenAIAdapter
from src.artifacts.adr_writer import write_adr
from src.artifacts.decision_matrix_writer import write_decision_matrix
from src.artifacts.writers import write_requirements
from src.gates.parsers import extract_json
from src.utils.io import read_text, write_json, write_text


class ArchitecturePipeline:
    def __init__(self, mode: str, base_dir: Path) -> None:
        self.mode = mode
        self.base_dir = base_dir
        self.schemas_dir = base_dir / "schemas"
        self.prompts_dir = base_dir / "configs" / "prompts"

    def run(self, brief_path: Path, run_dir: Path) -> Dict[str, Dict]:
        brief = read_text(brief_path)
        raw_dir = run_dir / "raw"
        artifacts_dir = run_dir / "artifacts"
        adrs_dir = artifacts_dir / "adrs"
        raw_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        adrs_dir.mkdir(parents=True, exist_ok=True)

        gemini = self._adapter("gemini")
        chatgpt = self._adapter("chatgpt")

        candidates = self._run_turn(
            gemini,
            "architecture_gemini",
            "architecture_candidates.schema.json",
            brief,
            raw_dir / "turn2_gemini_candidates.txt",
            artifacts_dir / "turn2_candidates.json",
        )

        normalized = self._run_turn(
            chatgpt,
            "architecture_chatgpt_normalize",
            "normalized_requirements.schema.json",
            brief,
            raw_dir / "turn3_chatgpt_normalize.txt",
            artifacts_dir / "turn3_normalized.json",
        )

        cross_review = self._run_turn(
            gemini,
            "architecture_gemini_cross_review",
            "normalized_requirements.schema.json",
            json.dumps(normalized),
            raw_dir / "turn4_gemini_cross_review.txt",
            artifacts_dir / "turn4_cross_review.json",
        )

        scoring = self._run_turn(
            chatgpt,
            "architecture_chatgpt_scoring",
            "scoring_result.schema.json",
            json.dumps({"candidates": candidates, "requirements": cross_review}),
            raw_dir / "turn5_chatgpt_scoring.txt",
            artifacts_dir / "turn5_scoring.json",
        )

        adr = self._run_turn(
            chatgpt,
            "adr_generation_chatgpt",
            "adr.schema.json",
            json.dumps(scoring),
            raw_dir / "turn5_chatgpt_adr.txt",
            artifacts_dir / "turn5_adr.json",
        )

        write_requirements(artifacts_dir / "requirements.md", normalized)
        write_decision_matrix(artifacts_dir / "decision_matrix.csv", candidates, scoring)
        write_adr(adrs_dir / f"{adr['adr_id']}.md", adr)
        self._write_architecture_summary(
            artifacts_dir / "architecture_summary.md", candidates, scoring
        )

        return {
            "candidates": candidates,
            "normalized": normalized,
            "cross_review": cross_review,
            "scoring": scoring,
            "adr": adr,
        }

    def _adapter(self, provider: str) -> LLMAdapter:
        if self.mode == "mock":
            return MockAdapter()
        if provider == "gemini":
            return GeminiAdapter()
        return OpenAIAdapter()

    def _run_turn(
        self,
        adapter: LLMAdapter,
        prompt_name: str,
        schema_name: str,
        brief: str,
        raw_path: Path,
        parsed_path: Path,
    ) -> Dict:
        prompt = read_text(self.prompts_dir / f"{prompt_name}.md")
        full_prompt = f"{prompt}\n\nINPUT:\n{brief}\n"
        response = adapter.complete(full_prompt)
        write_text(raw_path, response.raw_text)
        parsed = extract_json(response.raw_text)
        if schema_name == "architecture_candidates.schema.json":
            # Normalize candidate IDs to strings to satisfy strict schema requirements.
            for candidate in parsed.get("candidates", []):
                candidate_id = candidate.get("id")
                if candidate_id is not None and not isinstance(candidate_id, str):
                    candidate["id"] = str(candidate_id)
        if prompt_name == "architecture_gemini_cross_review":
            if not isinstance(parsed, dict):
                raise ValueError("Turn4 cross-review output must be a JSON object.")
            allowed_keys = {
                "issues",
                "risks",
                "suggestions",
                "suggested_requirements",
                "feedback",
                "review",
                "findings",
                "normalized_requirements",
                "requirements",
            }
            if not any(key in parsed for key in allowed_keys):
                keys = ", ".join(sorted(parsed.keys()))
                raise ValueError(
                    "Turn4 cross-review output must include at least one expected key. "
                    f"Found keys: {keys}"
                )
        else:
            schema = self._load_schema(schema_name)
            validate(instance=parsed, schema=schema)
        write_json(parsed_path, parsed)
        return parsed

    def _write_architecture_summary(
        self, path: Path, candidates: Dict, scoring: Dict
    ) -> None:
        score_map = {item["candidate_id"]: item for item in scoring.get("scores", [])}
        ordered: List[dict] = sorted(
            candidates.get("candidates", []),
            key=lambda item: score_map.get(item["id"], {}).get("total_score", 0),
            reverse=True,
        )
        lines = ["# Architecture Summary", ""]
        for index, candidate in enumerate(ordered, start=1):
            score = score_map.get(candidate["id"], {})
            total = score.get("total_score", "n/a")
            lines.extend(
                [
                    f"## {index}. {candidate['name']} ({candidate['id']})",
                    "",
                    candidate.get("summary", ""),
                    "",
                    f"Total score: {total}",
                    "",
                    "### Pros",
                    *[f"- {item}" for item in candidate.get("pros", [])],
                    "",
                    "### Cons",
                    *[f"- {item}" for item in candidate.get("cons", [])],
                    "",
                    "### Risks",
                    *[f"- {item}" for item in candidate.get("risks", [])],
                    "",
                ]
            )
        write_text(path, "\n".join(lines).strip() + "\n")

    def _load_schema(self, name: str) -> Dict:
        return json.loads(read_text(self.schemas_dir / name))
