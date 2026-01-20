from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Dict, Optional

from jsonschema import validate

from src.adapters.gemini_adapter import GeminiAdapter
from src.adapters.llm_base import LLMAdapter
from src.adapters.mock_adapter import MockAdapter
from src.adapters.openai_adapter import OpenAIAdapter
from src.gates.parsers import extract_json
from src.gates.runner import is_done, run_commands
from src.utils.io import read_text, write_json, write_text


class CodePipeline:
    def __init__(self, mode: str, base_dir: Path) -> None:
        self.mode = mode
        self.base_dir = base_dir
        self.schemas_dir = base_dir / "schemas"
        self.prompts_dir = base_dir / "configs" / "prompts"

    def run(
        self,
        run_dir: Path,
        commands: Iterable[str],
        brief_path: Path,
        inputs_from_run: Optional[str],
    ) -> Dict[str, object]:
        quality_dir = run_dir / "quality_gates"
        raw_dir = run_dir / "raw"
        artifacts_dir = run_dir / "artifacts"
        quality_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        brief = read_text(brief_path)
        previous_artifacts = self._load_previous_artifacts(inputs_from_run)

        chatgpt = self._adapter("chatgpt")
        gemini = self._adapter("gemini")

        task_payload = {
            "brief": brief,
            "requirements": previous_artifacts.get("requirements"),
            "architecture": previous_artifacts.get("architecture"),
        }
        task_prompt = read_text(self.prompts_dir / "code_task_chatgpt.md")
        task_response = chatgpt.complete(f"{task_prompt}\n\nINPUT:\n{json.dumps(task_payload)}\n")
        task_raw_path = raw_dir / "turnc1_chatgpt_code_task.txt"
        write_text(task_raw_path, task_response.raw_text)
        task_json = extract_json(task_response.raw_text)
        schema = self._load_schema("normalized_requirements.schema.json")
        validate(instance=task_json, schema=schema)
        write_json(artifacts_dir / "turnc1_code_tasks.json", task_json)

        patch_path = artifacts_dir / "proposed_patch.diff"
        write_text(patch_path, "# Patch placeholder generated in mock-first mode.\n")

        audit_prompt = read_text(self.prompts_dir / "code_audit_gemini.md")
        audit_response = gemini.complete(f"{audit_prompt}\n\nINPUT:\n{patch_path}\n")
        gemini_raw = raw_dir / "turnc2_gemini_audit.txt"
        write_text(gemini_raw, audit_response.raw_text)
        audit = extract_json(audit_response.raw_text)
        self._validate_audit(audit)
        write_json(artifacts_dir / "audit_report.json", audit)

        gpt_audit_prompt = read_text(self.prompts_dir / "code_audit_chatgpt.md")
        gpt_audit_response = chatgpt.complete(f"{gpt_audit_prompt}\n\nINPUT:\n{patch_path}\n")
        gpt_raw = raw_dir / "turnc3_chatgpt_audit.txt"
        write_text(gpt_raw, gpt_audit_response.raw_text)
        gpt_audit = extract_json(gpt_audit_response.raw_text)
        self._validate_audit(gpt_audit)
        write_json(artifacts_dir / "audit_report_chatgpt.json", gpt_audit)

        test_results = run_commands(commands, quality_dir)
        done = is_done(test_results, audit)

        return {
            "done": done,
            "audit": audit,
            "audit_chatgpt": gpt_audit,
            "tests": [result.return_code for result in test_results],
        }

    def _adapter(self, provider: str) -> LLMAdapter:
        if self.mode == "mock":
            return MockAdapter()
        if provider == "gemini":
            return GeminiAdapter()
        return OpenAIAdapter()

    def _load_schema(self, name: str) -> Dict:
        return json.loads(read_text(self.schemas_dir / name))

    def _validate_audit(self, audit: Dict) -> None:
        schema = self._load_schema("audit_report.schema.json")
        for finding in audit.get("findings", []):
            if not finding.get("location"):
                finding["location"] = "unknown"
        validate(instance=audit, schema=schema)

    def _load_previous_artifacts(self, run_id: Optional[str]) -> Dict[str, object]:
        if not run_id:
            return {}
        artifacts_dir = self.base_dir / "runs" / run_id / "artifacts"
        requirements_path = artifacts_dir / "requirements.json"
        if not requirements_path.exists():
            requirements_path = artifacts_dir / "turn3_normalized.json"
        requirements = None
        if requirements_path.exists():
            requirements = json.loads(read_text(requirements_path))

        scoring_path = artifacts_dir / "turn5_scoring.json"
        candidates_path = artifacts_dir / "turn2_candidates.json"
        architecture = None
        if scoring_path.exists() and candidates_path.exists():
            scoring = json.loads(read_text(scoring_path))
            candidates = json.loads(read_text(candidates_path))
            selected_id = scoring.get("selected_id")
            for candidate in candidates.get("candidates", []):
                if candidate.get("id") == selected_id:
                    architecture = candidate
                    break

        return {
            "requirements": requirements,
            "architecture": architecture,
        }
