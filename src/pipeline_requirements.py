from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from jsonschema import validate

from src.adapters.gemini_adapter import GeminiAdapter
from src.adapters.llm_base import LLMAdapter
from src.adapters.mock_adapter import MockAdapter
from src.adapters.openai_adapter import OpenAIAdapter
from src.artifacts.adr_writer import write_adr
from src.artifacts.writers import write_requirements
from src.gates.parsers import extract_json
from src.utils.io import read_text, write_json, write_text


class RequirementsPipeline:
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

        gaps = self._run_turn(
            gemini,
            "requirements_gemini",
            None,
            brief,
            raw_dir / "turnr1_gemini_gaps.txt",
            artifacts_dir / "turnr1_gemini_gaps.json",
        )

        normalized = self._run_turn(
            chatgpt,
            "architecture_chatgpt_normalize",
            "normalized_requirements.schema.json",
            json.dumps(gaps),
            raw_dir / "turnr2_chatgpt_normalize.txt",
            artifacts_dir / "requirements.json",
        )

        cross_review = self._run_turn(
            gemini,
            "architecture_gemini_cross_review",
            None,
            json.dumps(normalized),
            raw_dir / "turnr3_gemini_cross_review.txt",
            artifacts_dir / "turnr3_cross_review.json",
        )

        adr = self._run_turn(
            chatgpt,
            "adr_generation_chatgpt",
            "adr.schema.json",
            json.dumps({"requirements": normalized, "review": cross_review}),
            raw_dir / "turnr4_chatgpt_adr.txt",
            artifacts_dir / "turnr4_adr.json",
        )

        write_requirements(artifacts_dir / "requirements.md", normalized)
        write_adr(adrs_dir / f"{adr['adr_id']}.md", adr)

        return {
            "gaps": gaps,
            "normalized": normalized,
            "cross_review": cross_review,
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
        schema_name: str | None,
        brief: str,
        raw_path: Path,
        parsed_path: Path,
    ) -> Dict:
        prompt = read_text(self.prompts_dir / f"{prompt_name}.md")
        full_prompt = f"{prompt}\n\nINPUT:\n{brief}\n"
        response = adapter.complete(full_prompt)
        write_text(raw_path, response.raw_text)
        parsed = extract_json(response.raw_text)
        if schema_name == "normalized_requirements.schema.json" and isinstance(parsed, dict):
            if "requirements" not in parsed and {"id", "text", "priority"}.issubset(parsed.keys()):
                parsed = {
                    "requirements": [parsed],
                    "assumptions": [],
                    "constraints": [],
                }
        if schema_name:
            schema = self._load_schema(schema_name)
            validate(instance=parsed, schema=schema)
        else:
            if not isinstance(parsed, dict):
                raise ValueError("Requirements pipeline output must be a JSON object.")
        write_json(parsed_path, parsed)
        return parsed

    def _load_schema(self, name: str) -> Dict:
        return json.loads(read_text(self.schemas_dir / name))
