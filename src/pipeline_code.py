from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Dict

from jsonschema import validate

from src.adapters.gemini_adapter import GeminiAdapter
from src.adapters.llm_base import LLMAdapter
from src.adapters.mock_adapter import MockAdapter
from src.gates.parsers import extract_json
from src.gates.runner import is_done, run_commands
from src.utils.io import read_text, write_json, write_text


class CodePipeline:
    def __init__(self, mode: str, base_dir: Path) -> None:
        self.mode = mode
        self.base_dir = base_dir
        self.schemas_dir = base_dir / "schemas"
        self.prompts_dir = base_dir / "configs" / "prompts"

    def run(self, run_dir: Path, commands: Iterable[str]) -> Dict[str, object]:
        quality_dir = run_dir / "quality_gates"
        raw_dir = run_dir / "raw"
        artifacts_dir = run_dir / "artifacts"
        quality_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        test_results = run_commands(commands, quality_dir)

        adapter = self._adapter()
        prompt = read_text(self.prompts_dir / "code_audit_gemini.md")
        response = adapter.complete(prompt)
        raw_path = raw_dir / "code_audit_gemini.txt"
        write_text(raw_path, response.raw_text)
        audit = extract_json(response.raw_text)
        schema = self._load_schema("audit_report.schema.json")
        validate(instance=audit, schema=schema)
        write_json(artifacts_dir / "audit_report.json", audit)

        done = is_done(test_results, audit)
        return {"done": done, "audit": audit, "tests": [r.return_code for r in test_results]}

    def _adapter(self) -> LLMAdapter:
        if self.mode == "mock":
            return MockAdapter()
        return GeminiAdapter()

    def _load_schema(self, name: str) -> Dict:
        return json.loads(read_text(self.schemas_dir / name))
