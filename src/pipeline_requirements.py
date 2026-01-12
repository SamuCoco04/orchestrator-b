from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Tuple

from jsonschema import validate

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

        lead_prompt = read_text(self.prompts_dir / "requirements_chatgpt_lead.md")
        lead_full_prompt = f"{lead_prompt}\n\nINPUT:\n{brief}\n"
        lead_prompt_path = raw_dir / "turnr1_chatgpt_lead_prompt.txt"
        write_text(lead_prompt_path, lead_full_prompt)
        lead_response = chatgpt.complete(lead_full_prompt)
        lead_response_path = raw_dir / "turnr1_chatgpt_lead_response.txt"
        write_text(lead_response_path, lead_response.raw_text)
        self._write_usage(raw_dir / "turnr1_chatgpt_lead_usage.json", lead_response)

        review_json, requirements_json = self._parse_lead_output(lead_response.raw_text)
        review_schema = self._load_schema("requirements_review.schema.json")
        validate(instance=review_json, schema=review_schema)
        requirements_schema = self._load_schema("normalized_requirements.schema.json")
        validate(instance=requirements_json, schema=requirements_schema)

        write_json(artifacts_dir / "requirements_review.json", review_json)
        write_json(artifacts_dir / "requirements.json", requirements_json)

        cross_prompt = read_text(self.prompts_dir / "requirements_gemini_cross_review.md")
        cross_payload = {
            "brief": brief,
            "review": review_json,
            "requirements": requirements_json,
        }
        cross_full_prompt = f"{cross_prompt}\n\nINPUT:\n{json.dumps(cross_payload)}\n"
        cross_prompt_path = raw_dir / "turnr2_gemini_cross_review_prompt.txt"
        write_text(cross_prompt_path, cross_full_prompt)
        cross_response = gemini.complete(cross_full_prompt)
        cross_response_path = raw_dir / "turnr2_gemini_cross_review_response.txt"
        write_text(cross_response_path, cross_response.raw_text)
        self._write_usage(raw_dir / "turnr2_gemini_cross_review_usage.json", cross_response)

        cross_review = extract_json(cross_response.raw_text)
        if not isinstance(cross_review, dict):
            raise ValueError("Turn R2 cross-review output must be a JSON object.")
        write_json(artifacts_dir / "turnr2_gemini_cross_review.json", cross_review)

        adr_prompt = read_text(self.prompts_dir / "requirements_chatgpt_adr.md")
        adr_payload = {
            "requirements": requirements_json,
            "review": review_json,
            "cross_review": cross_review,
        }
        adr_full_prompt = f"{adr_prompt}\n\nINPUT:\n{json.dumps(adr_payload)}\n"
        adr_prompt_path = raw_dir / "turnr3_chatgpt_adr_prompt.txt"
        write_text(adr_prompt_path, adr_full_prompt)
        adr_response = chatgpt.complete(adr_full_prompt)
        adr_response_path = raw_dir / "turnr3_chatgpt_adr_response.txt"
        write_text(adr_response_path, adr_response.raw_text)
        self._write_usage(raw_dir / "turnr3_chatgpt_adr_usage.json", adr_response)

        adr = extract_json(adr_response.raw_text)
        adr_schema = self._load_schema("adr.schema.json")
        validate(instance=adr, schema=adr_schema)
        write_json(artifacts_dir / "turnr3_chatgpt_adr.json", adr)

        write_requirements(artifacts_dir / "requirements.md", requirements_json)
        write_adr(adrs_dir / f"{adr['adr_id']}.md", adr)

        return {
            "review": review_json,
            "requirements": requirements_json,
            "cross_review": cross_review,
            "adr": adr,
        }

    def _adapter(self, provider: str) -> LLMAdapter:
        if self.mode == "mock":
            return MockAdapter()
        if provider == "gemini":
            return GeminiAdapter()
        return OpenAIAdapter()

    def _parse_lead_output(self, raw_text: str) -> Tuple[Dict, Dict]:
        label_pattern = (
            r"REVIEW_JSON\s*:?[\r\n]+(.*?)"
            r"REQUIREMENTS_JSON\s*:?[\r\n]+(.*)"
        )
        match = re.search(label_pattern, raw_text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            review_text, requirements_text = match.groups()
            review_json = extract_json(review_text)
            requirements_json = extract_json(requirements_text)
            return review_json, requirements_json

        decoder = json.JSONDecoder()
        found: list[dict] = []
        keys_seen: list[list[str]] = []
        idx = 0
        while idx < len(raw_text) and len(found) < 2:
            start = self._find_next_json_start(raw_text, idx)
            if start == -1:
                break
            try:
                parsed, end = decoder.raw_decode(raw_text[start:])
                found.append(parsed)
                if isinstance(parsed, dict):
                    keys_seen.append(sorted(parsed.keys()))
                idx = start + end
            except json.JSONDecodeError:
                idx = start + 1

        if len(found) >= 2:
            return found[0], found[1]

        snippet = raw_text.strip().replace("\n", " ")
        snippet = (snippet[:300] + "...") if len(snippet) > 300 else snippet
        raise ValueError(
            "Lead output must include REVIEW_JSON and REQUIREMENTS_JSON blocks. "
            f"Detected JSON keys: {keys_seen}. Snippet: {snippet}"
        )

    def _find_next_json_start(self, text: str, start: int) -> int:
        for idx in range(start, len(text)):
            if text[idx] in "{[":
                return idx
        return -1

    def _write_usage(self, path: Path, response: LLMResponse) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            write_json(path, usage)

    def _load_schema(self, name: str) -> Dict:
        return json.loads(read_text(self.schemas_dir / name))
