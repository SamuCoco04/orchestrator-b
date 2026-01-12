from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from jsonschema import ValidationError, validate

from src.adapters.gemini_adapter import GeminiAdapter
from src.adapters.llm_base import LLMAdapter, LLMResponse
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

        requirements_prompt = read_text(self.prompts_dir / "requirements_chatgpt_requirements.md")
        requirements_full_prompt = f"{requirements_prompt}\n\nINPUT:\n{brief}\n"
        requirements_prompt_path = raw_dir / "turnr1_chatgpt_requirements_prompt.txt"
        write_text(requirements_prompt_path, requirements_full_prompt)
        requirements_response = chatgpt.complete(requirements_full_prompt)
        requirements_response_path = raw_dir / "turnr1_chatgpt_requirements_response.txt"
        write_text(requirements_response_path, requirements_response.raw_text)
        self._write_usage(raw_dir / "turnr1_chatgpt_requirements_usage.json", requirements_response)

        requirements_json = extract_json(requirements_response.raw_text)
        requirements_schema = self._load_schema("normalized_requirements.schema.json")
        validate(instance=requirements_json, schema=requirements_schema)
        write_json(artifacts_dir / "requirements.json", requirements_json)

        review_prompt = read_text(self.prompts_dir / "requirements_chatgpt_review.md")
        review_payload = {"brief": brief, "requirements": requirements_json}
        review_full_prompt = f"{review_prompt}\n\nINPUT:\n{json.dumps(review_payload)}\n"
        review_prompt_path = raw_dir / "turnr2_chatgpt_review_prompt.txt"
        write_text(review_prompt_path, review_full_prompt)
        review_response = chatgpt.complete(review_full_prompt)
        review_response_path = raw_dir / "turnr2_chatgpt_review_response.txt"
        write_text(review_response_path, review_response.raw_text)
        self._write_usage(raw_dir / "turnr2_chatgpt_review_usage.json", review_response)

        review_json = extract_json(review_response.raw_text)
        review_schema = self._load_schema("requirements_review.schema.json")
        try:
            validate(instance=review_json, schema=review_schema)
        except ValidationError:
            if self.mode != "mock":
                raise
            review_json = self._mock_review()
            validate(instance=review_json, schema=review_schema)
        write_json(artifacts_dir / "requirements_review.json", review_json)

        cross_prompt = read_text(self.prompts_dir / "requirements_gemini_cross_review.md")
        cross_payload = {
            "brief": brief,
            "requirements": requirements_json,
            "review": review_json,
        }
        cross_full_prompt = f"{cross_prompt}\n\nINPUT:\n{json.dumps(cross_payload)}\n"
        cross_prompt_path = raw_dir / "turnr3_gemini_cross_review_prompt.txt"
        write_text(cross_prompt_path, cross_full_prompt)
        cross_response = gemini.complete(cross_full_prompt)
        cross_response_path = raw_dir / "turnr3_gemini_cross_review_response.txt"
        write_text(cross_response_path, cross_response.raw_text)
        self._write_usage(raw_dir / "turnr3_gemini_cross_review_usage.json", cross_response)

        cross_review = extract_json(cross_response.raw_text)
        if not isinstance(cross_review, dict):
            raise ValueError("Turn R3 cross-review output must be a JSON object.")
        write_json(artifacts_dir / "turnr3_gemini_cross_review.json", cross_review)

        adr_prompt = read_text(self.prompts_dir / "requirements_chatgpt_adr.md")
        adr_payload = {
            "requirements": requirements_json,
            "review": review_json,
            "cross_review": cross_review,
        }
        adr_full_prompt = f"{adr_prompt}\n\nINPUT:\n{json.dumps(adr_payload)}\n"
        adr_prompt_path = raw_dir / "turnr4_chatgpt_adr_prompt.txt"
        write_text(adr_prompt_path, adr_full_prompt)
        adr_response = chatgpt.complete(adr_full_prompt)
        adr_response_path = raw_dir / "turnr4_chatgpt_adr_response.txt"
        write_text(adr_response_path, adr_response.raw_text)
        self._write_usage(raw_dir / "turnr4_chatgpt_adr_usage.json", adr_response)

        adr = extract_json(adr_response.raw_text)
        adr_schema = self._load_schema("adr.schema.json")
        validate(instance=adr, schema=adr_schema)
        write_json(artifacts_dir / "turnr4_chatgpt_adr.json", adr)

        write_requirements(artifacts_dir / "requirements.md", requirements_json)
        write_adr(adrs_dir / f"{adr['adr_id']}.md", adr)

        return {
            "requirements": requirements_json,
            "review": review_json,
            "cross_review": cross_review,
            "adr": adr,
        }

    def _adapter(self, provider: str) -> LLMAdapter:
        if self.mode == "mock":
            return MockAdapter()
        if provider == "gemini":
            return GeminiAdapter()
        return OpenAIAdapter()

    def _mock_review(self) -> Dict:
        return {
            "accepted": ["Add Erasmus Coordinator role"],
            "rejected": [],
            "issues": ["Clarify ownership of deadline tracking"],
            "missing": ["Define compliance checkpoints"],
            "rationale": ["Role clarity reduces process risk"],
        }

    def _write_usage(self, path: Path, response: LLMResponse) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            write_json(path, usage)

    def _load_schema(self, name: str) -> Dict:
        return json.loads(read_text(self.schemas_dir / name))
