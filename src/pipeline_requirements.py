from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from jsonschema import ValidationError, validate

from src.adapters.gemini_adapter import GeminiAdapter
from src.adapters.llm_base import LLMAdapter, LLMResponse
from src.adapters.mock_adapter import MockAdapter
from src.adapters.openai_adapter import OpenAIAdapter
from src.artifacts.adr_writer import write_adr
from src.artifacts.writers import write_requirements
from src.gates.parsers import extract_json
from src.utils.io import read_text, write_json, write_text


@dataclass
class RequirementsLimits:
    req_min: int
    req_max: int | None
    assumptions_min: int
    constraints_min: int
    roles_expected: List[str]
    coverage_areas: List[str]
    coverage_keywords: Dict[str, List[str]]
    min_per_area: int | None
    seed_requirements: List[str]
    requested_artifacts: List[str]
    artifact_token_budgets: Dict[str, int]


class RequirementsPipeline:
    def __init__(self, mode: str, base_dir: Path) -> None:
        self.mode = mode
        self.base_dir = base_dir
        self.schemas_dir = base_dir / "schemas"
        self.prompts_dir = base_dir / "configs" / "prompts"
        self._review_normalization_warnings: List[str] = []
        self._extraction_traces: List[str] = []
        self._repair_warnings: List[Dict] = []
        self._acceptance_warnings: List[str] = []
        self._section_warnings: Dict[str, List[str]] = {}
        self._requirements_warnings: List[Dict] = []
        self._list_repair_counts: Dict[str, int] = {
            "requirements": 0,
            "assumptions": 0,
            "constraints": 0,
            "moved": 0,
        }
        self._artifact_repair_counts: Dict[str, int] = {}
        self._delta_retry_counts: Dict[str, int] = {}
        self._artifact_validation: Dict[str, str] = {}

    def run(
        self, brief_path: Path, run_dir: Path, artifact: str = "requirements"
    ) -> Dict[str, Dict]:
        self._review_normalization_warnings = []
        self._extraction_traces = []
        self._repair_warnings = []
        self._acceptance_warnings = []
        self._section_warnings = {}
        self._requirements_warnings = []
        self._list_repair_counts = {
            "requirements": 0,
            "assumptions": 0,
            "constraints": 0,
            "moved": 0,
        }
        self._artifact_repair_counts = {}
        self._delta_retry_counts = {}
        self._artifact_validation = {}
        raw_brief = read_text(brief_path)
        frontmatter, brief = self._parse_frontmatter(raw_brief)
        limits = self._limits_from_frontmatter(frontmatter)

        raw_dir = run_dir / "raw"
        artifacts_dir = run_dir / "artifacts"
        raw_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        artifact_key = artifact.strip().lower()
        if artifact_key not in self._artifact_configs():
            raise ValueError(f"Unsupported artifact: {artifact}")

        gemini = self._adapter("gemini")
        chatgpt = self._adapter("chatgpt")

        (
            payload,
            warnings,
            retry_count,
            missing_fields,
            responses,
            summary,
        ) = self._run_single_artifact(
            artifact_key,
            brief,
            limits,
            chatgpt,
            gemini,
            raw_dir,
            artifacts_dir,
        )
        self._write_single_run_summary(
            artifacts_dir,
            artifact_key,
            payload,
            warnings,
            retry_count,
            missing_fields,
            responses,
            summary,
        )
        return {artifact_key: payload}

    def _artifact_configs(self) -> Dict[str, Dict]:
        return {
            "requirements": {
                "lead_prompt": "requirements_lead.md",
                "apply_prompt": "requirements_apply.md",
                "draft_label": "REQUIREMENTS_JSON",
                "final_label": "FINAL_REQUIREMENTS_JSON",
                "schema": "normalized_requirements.schema.json",
                "expected_keys": {"requirements", "assumptions", "constraints"},
                "default_budget": 2400,
            },
            "business_rules": {
                "lead_prompt": "business_rules_lead.md",
                "apply_prompt": "business_rules_apply.md",
                "draft_label": "BUSINESS_RULES_JSON",
                "final_label": "FINAL_BUSINESS_RULES_JSON",
                "schema": "business_rules.schema.json",
                "expected_keys": {"rules"},
                "default_budget": 1600,
            },
            "workflows": {
                "lead_prompt": "workflows_lead.md",
                "apply_prompt": "workflows_apply.md",
                "draft_label": "WORKFLOWS_JSON",
                "final_label": "FINAL_WORKFLOWS_JSON",
                "schema": "workflows.schema.json",
                "expected_keys": {"workflows"},
                "default_budget": 2000,
            },
            "domain_model": {
                "lead_prompt": "domain_model_lead.md",
                "apply_prompt": "domain_model_apply.md",
                "draft_label": "DOMAIN_MODEL_JSON",
                "final_label": "FINAL_DOMAIN_MODEL_JSON",
                "schema": "domain_model.schema.json",
                "expected_keys": {"entities", "relationships"},
                "default_budget": 1600,
            },
            "mvp_scope": {
                "lead_prompt": "mvp_scope_lead.md",
                "apply_prompt": "mvp_scope_apply.md",
                "draft_label": "MVP_SCOPE_JSON",
                "final_label": "FINAL_MVP_SCOPE_JSON",
                "schema": "mvp_scope.schema.json",
                "expected_keys": {"in_scope", "out_of_scope"},
                "default_budget": 1200,
            },
            "acceptance_criteria": {
                "lead_prompt": "acceptance_criteria_lead.md",
                "apply_prompt": "acceptance_criteria_apply.md",
                "draft_label": "ACCEPTANCE_CRITERIA_JSON",
                "final_label": "FINAL_ACCEPTANCE_CRITERIA_JSON",
                "schema": "acceptance_criteria.schema.json",
                "expected_keys": {"criteria"},
                "default_budget": 2000,
            },
        }

    def _artifact_token_budget(self, artifact: str, limits: RequirementsLimits) -> int:
        config = self._artifact_configs()[artifact]
        default_budget = config["default_budget"]
        value = limits.artifact_token_budgets.get(artifact)
        if isinstance(value, int) and value > 0:
            return value
        return default_budget

    @contextmanager
    def _with_max_output_tokens(self, max_tokens: int) -> None:
        original = os.getenv("ORCH_MAX_OUTPUT_TOKENS")
        os.environ["ORCH_MAX_OUTPUT_TOKENS"] = str(max_tokens)
        try:
            yield
        finally:
            if original is None:
                os.environ.pop("ORCH_MAX_OUTPUT_TOKENS", None)
            else:
                os.environ["ORCH_MAX_OUTPUT_TOKENS"] = original

    def _run_single_artifact(
        self,
        artifact: str,
        brief: str,
        limits: RequirementsLimits,
        chatgpt: LLMAdapter,
        gemini: LLMAdapter,
        raw_dir: Path,
        artifacts_dir: Path,
    ) -> tuple[Dict, List[Dict], int, List[str], List[LLMResponse], Dict]:
        config = self._artifact_configs()[artifact]
        max_tokens = self._artifact_token_budget(artifact, limits)
        responses: List[LLMResponse] = []
        summary: Dict[str, object] = {}

        lead_template = read_text(self.prompts_dir / config["lead_prompt"])
        lead_prompt = self._render_prompt(lead_template, limits)
        lead_payload = {"brief": brief}
        lead_full_prompt = f"{lead_prompt}\n\nINPUT:\n{json.dumps(lead_payload)}\n"
        write_text(raw_dir / f"{artifact}_draft_prompt.txt", lead_full_prompt)
        with self._with_max_output_tokens(max_tokens):
            lead_response = chatgpt.complete(lead_full_prompt)
        responses.append(lead_response)
        write_text(raw_dir / f"{artifact}_draft_raw.txt", lead_response.raw_text)
        self._write_usage(raw_dir / f"{artifact}_draft_usage.json", lead_response)

        draft_payload = self._extract_wrapped_json(
            lead_response.raw_text,
            config["draft_label"],
            config["expected_keys"],
        )
        draft_payload, draft_warnings = self._repair_artifact_payload(
            artifact, draft_payload, stage="draft"
        )

        cross_review_prompt = self._artifact_cross_review_prompt(artifact)
        cross_payload = {"brief": brief, "artifact": draft_payload}
        cross_full_prompt = f"{cross_review_prompt}\n\nINPUT:\n{json.dumps(cross_payload)}\n"
        write_text(raw_dir / f"{artifact}_cross_review_prompt.txt", cross_full_prompt)
        cross_response = gemini.complete(cross_full_prompt)
        responses.append(cross_response)
        write_text(raw_dir / f"{artifact}_cross_review_raw.txt", cross_response.raw_text)
        self._write_usage(raw_dir / f"{artifact}_cross_review_usage.json", cross_response)
        cross_review = self._safe_extract_json(cross_response.raw_text)

        apply_template = read_text(self.prompts_dir / config["apply_prompt"])
        apply_prompt = self._render_prompt(apply_template, limits)
        apply_payload = {
            "brief": brief,
            "draft": draft_payload,
            "cross_review": cross_review,
        }
        if artifact == "requirements":
            apply_payload["targets"] = self._requirements_targets_payload(limits)
            apply_payload["gemini_review"] = cross_review
            apply_payload["gemini_review_text"] = cross_response.raw_text
        apply_instruction = ""
        if artifact == "requirements":
            apply_instruction = (
                "\n\nYou may add NEW requirements to meet targets and missing coverage."
            )
        apply_full_prompt = f"{apply_prompt}{apply_instruction}\n\nINPUT:\n{json.dumps(apply_payload)}\n"
        write_text(raw_dir / f"{artifact}_apply_prompt.txt", apply_full_prompt)
        with self._with_max_output_tokens(max_tokens):
            apply_response = chatgpt.complete(apply_full_prompt)
        responses.append(apply_response)
        write_text(raw_dir / f"{artifact}_apply_raw.txt", apply_response.raw_text)
        self._write_usage(raw_dir / f"{artifact}_apply_usage.json", apply_response)

        final_payload = self._extract_wrapped_json(
            apply_response.raw_text,
            config["final_label"],
            config["expected_keys"],
        )
        final_payload, final_warnings = self._repair_artifact_payload(
            artifact, final_payload, stage="apply"
        )

        warnings = draft_warnings + final_warnings
        if warnings:
            self._artifact_repair_counts[artifact] = len(warnings)
            write_json(
                artifacts_dir / f"{artifact}_warnings.json",
                {"warnings": warnings},
            )

        retry_count = 0
        add_only_used = False
        id_normalized = False
        missing_coverage_areas: List[str] = []
        missing_fields: List[str] = []
        schema = self._load_schema(config["schema"])
        if artifact == "requirements":
            final_payload, id_normalized = self._normalize_requirement_ids(
                final_payload
            )
            final_payload, missing_coverage_areas, add_only_used = self._ensure_requirements_targets(
                brief=brief,
                limits=limits,
                payload=final_payload,
                adapter=chatgpt,
                raw_dir=raw_dir,
                max_tokens=max_tokens,
            )
            summary.update(
                {
                    "target_min_items": limits.req_min,
                    "actual_count": len(final_payload.get("requirements", [])),
                    "missing_coverage_areas": missing_coverage_areas,
                    "add_only_retry_used": add_only_used,
                    "id_normalized": id_normalized,
                }
            )
        try:
            validate(instance=final_payload, schema=schema)
            self._artifact_validation[artifact] = "valid"
        except ValidationError as exc:
            missing_fields = self._extract_missing_fields(exc)
            retry_payload = {
                "brief": brief,
                "draft": draft_payload,
                "cross_review": cross_review,
                "current": final_payload,
                "validation_errors": [str(exc)],
            }
            retry_full_prompt = (
                f"{apply_prompt}\n\nFix ONLY missing/invalid fields in the current payload. "
                "Do not rewrite or regenerate unrelated content.\n\nINPUT:\n"
                f"{json.dumps(retry_payload)}\n"
            )
            write_text(raw_dir / f"{artifact}_apply_retry_prompt.txt", retry_full_prompt)
            with self._with_max_output_tokens(max_tokens):
                retry_response = chatgpt.complete(retry_full_prompt)
            responses.append(retry_response)
            write_text(raw_dir / f"{artifact}_apply_retry_raw.txt", retry_response.raw_text)
            self._write_usage(raw_dir / f"{artifact}_apply_retry_usage.json", retry_response)
            retry_count += 1
            try:
                retry_payload_json = self._extract_wrapped_json(
                    retry_response.raw_text,
                    config["final_label"],
                    config["expected_keys"],
                )
                retry_payload_json, retry_warnings = self._repair_artifact_payload(
                    artifact, retry_payload_json, stage="apply_retry"
                )
                if retry_warnings:
                    warnings.extend(retry_warnings)
                    self._artifact_repair_counts[artifact] = len(warnings)
                    write_json(
                        artifacts_dir / f"{artifact}_warnings.json",
                        {"warnings": warnings},
                    )
                validate(instance=retry_payload_json, schema=schema)
                final_payload = retry_payload_json
                self._artifact_validation[artifact] = "valid"
            except ValidationError as retry_exc:
                missing_fields = self._extract_missing_fields(retry_exc)
                self._artifact_validation[artifact] = "invalid"
        except Exception:
            self._artifact_validation[artifact] = "invalid"

        self._write_artifact_outputs(artifact, final_payload, artifacts_dir)
        return final_payload, warnings, retry_count, missing_fields, responses, summary

    def _safe_extract_json(self, raw_text: str) -> Dict:
        try:
            parsed = extract_json(raw_text)
        except ValueError:
            return {"notes": raw_text.strip()}
        if isinstance(parsed, dict):
            return parsed
        return {"notes": raw_text.strip()}

    def _extract_wrapped_json(
        self, raw_text: str, label: str, expected_keys: set[str]
    ) -> Dict:
        payload = self._safe_extract_json(raw_text)
        if label in payload and isinstance(payload[label], dict):
            payload = payload[label]
        if not isinstance(payload, dict):
            raise ValueError(f"{label} payload must be a JSON object.")
        missing = expected_keys.difference(payload.keys())
        if missing:
            raise ValueError(f"{label} missing keys: {', '.join(sorted(missing))}")
        return payload

    def _artifact_cross_review_prompt(self, artifact: str) -> str:
        return (
            "You are a critical reviewer. Identify ambiguity, missing details, "
            "edge cases, and contradictions in the artifact. Do NOT rewrite the artifact. "
            "Return JSON: {\"issues\":[],\"missing\":[],\"ambiguities\":[],"
            "\"edge_cases\":[],\"recommendations\":[]}."
        )

    def _repair_artifact_payload(
        self, artifact: str, payload: Dict, stage: str
    ) -> tuple[Dict, List[Dict]]:
        warnings: List[Dict] = []
        if artifact == "requirements":
            payload, repair_warnings = self._repair_requirements_payload(payload)
            warnings.extend(repair_warnings)
            start_len = len(self._requirements_warnings)
            payload, _ = self._normalize_requirements_payload(payload, stage=stage)
            warnings.extend(self._requirements_warnings[start_len:])
            return payload, warnings
        if artifact == "business_rules":
            return self._normalize_business_rules(payload), warnings
        if artifact == "workflows":
            repaired, notes = self._repair_workflows(payload)
            warnings.extend({"warning": note} for note in notes)
            return repaired, warnings
        if artifact == "domain_model":
            repaired, notes = self._repair_domain_model(payload)
            warnings.extend({"warning": note} for note in notes)
            return repaired, warnings
        if artifact == "mvp_scope":
            repaired, notes = self._repair_mvp_scope(payload)
            warnings.extend({"warning": note} for note in notes)
            return repaired, warnings
        return payload, warnings

    def _write_artifact_outputs(
        self, artifact: str, payload: Dict, artifacts_dir: Path
    ) -> None:
        write_json(artifacts_dir / f"{artifact}.json", payload)
        if artifact == "requirements":
            write_requirements(artifacts_dir / "requirements.md", payload)
        elif artifact == "business_rules":
            self._write_business_rules_markdown(artifacts_dir / "business_rules.md", payload)
        elif artifact == "workflows":
            self._write_workflows_markdown(artifacts_dir / "workflows.md", payload)
        elif artifact == "domain_model":
            self._write_domain_model_markdown(artifacts_dir / "domain_model.md", payload)
        elif artifact == "mvp_scope":
            self._write_mvp_scope_markdown(artifacts_dir / "mvp_scope.md", payload)
        elif artifact == "acceptance_criteria":
            self._write_acceptance_markdown(
                artifacts_dir / "acceptance_criteria.md", payload
            )

    def _extract_missing_fields(self, exc: ValidationError) -> List[str]:
        missing = []
        if exc.validator == "required":
            matches = re.findall(r"'([^']+)' is a required property", str(exc))
            missing.extend(matches)
        return missing

    def _artifact_count(self, artifact: str, payload: Dict) -> int:
        if artifact == "requirements":
            return len(payload.get("requirements", []))
        if artifact == "business_rules":
            return len(payload.get("rules", []))
        if artifact == "workflows":
            return len(payload.get("workflows", []))
        if artifact == "domain_model":
            return len(payload.get("entities", []))
        if artifact == "mvp_scope":
            return len(payload.get("in_scope", []))
        if artifact == "acceptance_criteria":
            return len(payload.get("criteria", []))
        return 0

    def _requirements_targets_payload(self, limits: RequirementsLimits) -> Dict:
        return {
            "target_min_items": limits.req_min,
            "target_max_items": limits.req_max,
            "min_assumptions": limits.assumptions_min,
            "min_constraints": limits.constraints_min,
            "coverage_areas": limits.coverage_areas,
            "coverage_keywords": limits.coverage_keywords,
            "min_per_area": limits.min_per_area,
        }

    def _coverage_keywords_for_area(self, limits: RequirementsLimits, area: str) -> List[str]:
        keywords = limits.coverage_keywords.get(area, [])
        if area.lower() not in [value.lower() for value in keywords]:
            return [area] + keywords
        return keywords

    def _missing_coverage_areas(self, payload: Dict, limits: RequirementsLimits) -> List[str]:
        if limits.min_per_area is None or not limits.coverage_areas:
            return []
        min_per_area = limits.min_per_area
        req_items = payload.get("requirements", [])
        texts = [str(item.get("text", "")).lower() for item in req_items if isinstance(item, dict)]
        missing: List[str] = []
        for area in limits.coverage_areas:
            keywords = self._coverage_keywords_for_area(limits, area)
            count = 0
            for text in texts:
                if any(keyword.lower() in text for keyword in keywords):
                    count += 1
            if count < min_per_area:
                missing.append(area)
        return missing

    def _ensure_requirements_targets(
        self,
        brief: str,
        limits: RequirementsLimits,
        payload: Dict,
        adapter: LLMAdapter,
        raw_dir: Path,
        max_tokens: int,
    ) -> tuple[Dict, List[str], bool]:
        missing_coverage = self._missing_coverage_areas(payload, limits)
        current_count = len(payload.get("requirements", []))
        missing_count = max(limits.req_min - current_count, 0)
        missing_assumptions = max(limits.assumptions_min - len(payload.get("assumptions", [])), 0)
        missing_constraints = max(limits.constraints_min - len(payload.get("constraints", [])), 0)
        if not any([missing_coverage, missing_count, missing_assumptions, missing_constraints]):
            return payload, missing_coverage, False

        retry_prompt = read_text(self.prompts_dir / "requirements_add_only_retry.md")
        retry_payload = {
            "brief": brief,
            "current_requirements": payload,
            "targets": self._requirements_targets_payload(limits),
            "missing_coverage_areas": missing_coverage,
            "missing_count": missing_count,
            "missing_assumptions": missing_assumptions,
            "missing_constraints": missing_constraints,
        }
        full_prompt = f"{retry_prompt}\n\nINPUT:\n{json.dumps(retry_payload)}\n"
        write_text(raw_dir / "requirements_add_only_retry_prompt.txt", full_prompt)
        with self._with_max_output_tokens(max_tokens):
            response = adapter.complete(full_prompt)
        write_text(raw_dir / "requirements_add_only_retry_raw.txt", response.raw_text)
        self._write_usage(raw_dir / "requirements_add_only_retry_usage.json", response)
        additions = self._extract_wrapped_json(
            response.raw_text,
            "REQUIREMENTS_JSON",
            {"requirements", "assumptions", "constraints"},
        )
        additions, _ = self._repair_artifact_payload("requirements", additions, stage="add_only")
        payload = self._merge_requirements_additions(payload, additions)
        payload, _ = self._normalize_requirement_ids(payload)
        missing_coverage = self._missing_coverage_areas(payload, limits)
        return payload, missing_coverage, True

    def _merge_requirements_additions(self, base: Dict, additions: Dict) -> Dict:
        merged = {
            "requirements": list(base.get("requirements", [])),
            "assumptions": list(base.get("assumptions", [])),
            "constraints": list(base.get("constraints", [])),
        }
        for item in additions.get("requirements", []):
            if isinstance(item, dict):
                merged["requirements"].append(item)
        for item in additions.get("assumptions", []):
            if isinstance(item, str):
                merged["assumptions"].append(item)
        for item in additions.get("constraints", []):
            if isinstance(item, str):
                merged["constraints"].append(item)
        return merged

    def _normalize_requirement_ids(self, payload: Dict) -> tuple[Dict, bool]:
        items = payload.get("requirements", [])
        if not isinstance(items, list):
            return payload, False
        normalized = False
        used_ids: set[str] = set()
        sequence = 1

        def next_id() -> str:
            nonlocal sequence
            while True:
                candidate = f"REQ-{sequence:03d}"
                sequence += 1
                if candidate not in used_ids:
                    used_ids.add(candidate)
                    return candidate

        normalized_items: List[Dict] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            current_id = item.get("id")
            needs_new = False
            if current_id is None:
                needs_new = True
            elif isinstance(current_id, int):
                needs_new = True
            elif isinstance(current_id, str):
                if current_id.strip().isdigit():
                    needs_new = True
                elif current_id in used_ids:
                    needs_new = True
            else:
                needs_new = True

            if needs_new:
                item["id"] = next_id()
                normalized = True
            else:
                used_ids.add(current_id)
            normalized_items.append(item)

        payload["requirements"] = normalized_items
        return payload, normalized

    def _write_single_run_summary(
        self,
        artifacts_dir: Path,
        artifact: str,
        payload: Dict,
        warnings: List[Dict],
        retry_count: int,
        missing_fields: List[str],
        responses: List[LLMResponse],
        summary: Dict[str, object],
    ) -> None:
        usage_totals = self._collect_usage_totals(artifacts_dir.parent / "raw", responses)
        lines = [
            "# Run Summary",
            "",
            f"- artifact: {artifact}",
            f"- item_count: {self._artifact_count(artifact, payload)}",
            f"- repairs_applied: {'yes' if warnings else 'no'}",
            f"- missing_fields: {', '.join(missing_fields) if missing_fields else 'none'}",
            f"- retry_count: {retry_count}",
        ]
        if artifact == "requirements" and summary:
            target_min_items = summary.get("target_min_items")
            actual_count = summary.get("actual_count")
            missing_coverage = summary.get("missing_coverage_areas", [])
            add_only_used = summary.get("add_only_retry_used")
            id_normalized = summary.get("id_normalized")
            lines.append(f"- target_min_items: {target_min_items}")
            lines.append(f"- actual_count: {actual_count}")
            if missing_coverage:
                lines.append(f"- missing_coverage_areas: {', '.join(missing_coverage)}")
            else:
                lines.append("- missing_coverage_areas: none")
            lines.append(f"- add_only_retry_used: {'yes' if add_only_used else 'no'}")
            lines.append(f"- id_normalized: {'yes' if id_normalized else 'no'}")
        if self._artifact_validation.get(artifact):
            lines.append(f"- validation: {self._artifact_validation[artifact]}")
        if usage_totals:
            lines.append(
                "- token_usage: "
                + ", ".join(f"{key}={value}" for key, value in usage_totals.items())
            )
        write_text(artifacts_dir / "run_summary.md", "\n".join(lines) + "\n")

    def _adapter(self, provider: str) -> LLMAdapter:
        if self.mode == "mock":
            return MockAdapter()
        if provider == "gemini":
            return GeminiAdapter()
        return OpenAIAdapter()

    def _run_apply(
        self,
        adapter: LLMAdapter,
        prompt: str,
        brief: str,
        draft: Dict,
        review: Dict,
        cross_review: Dict,
        limits: RequirementsLimits,
        raw_dir: Path,
        artifacts_dir: Path,
    ) -> tuple[Dict, Dict, bool]:
        failures: List[str] = []
        repairs_applied = False

        def write_normalized_if_repaired(
            final_payload: Dict, changelog_payload: Dict
        ) -> None:
            if repairs_applied:
                write_json(
                    raw_dir / "turnr4_final_requirements_normalized.json",
                    final_payload,
                )
                write_json(
                    raw_dir / "turnr4_changelog_normalized.json",
                    changelog_payload,
                )
        for attempt in range(2):
            instruction = ""
            if attempt == 1:
                instruction = (
                    "\n\nPrevious output failed gates: "
                    f"{', '.join(failures)}. Fix them now."
                )
            payload = {
                "brief": brief,
                "requirements": draft,
                "review": review,
                "cross_review": cross_review,
                "limits": self._limits_payload(limits),
            }
            full_prompt = f"{prompt}\n\nINPUT:\n{json.dumps(payload)}{instruction}\n"
            prompt_path = raw_dir / "turnr4_apply_prompt.txt"
            write_text(prompt_path, full_prompt)
            response = adapter.complete(full_prompt)
            write_text(raw_dir / "turnr4_apply_raw.txt", response.raw_text)
            self._write_usage(raw_dir / "turnr4_apply_usage.json", response)

            final_requirements = self._extract_marked_json(
                response.raw_text,
                "FINAL_REQUIREMENTS_JSON:",
                {"requirements", "assumptions", "constraints"},
            )
            final_requirements, repair_warnings = self._repair_requirements_payload(
                final_requirements
            )
            if repair_warnings:
                self._repair_warnings.extend(repair_warnings)
                write_json(
                    artifacts_dir / "repairs_warnings.json",
                    {"warnings": self._repair_warnings},
                )
            final_requirements, string_count = self._normalize_requirements_payload(
                final_requirements, stage="final"
            )
            if string_count > 2:
                retry_requirements = self._retry_requirements_only(
                    adapter,
                    brief,
                    limits,
                    raw_dir,
                    "requirements_apply_retry_requirements_only.md",
                    "turnr4_requirements_retry",
                    "FINAL_REQUIREMENTS_JSON:",
                    {"requirements", "assumptions", "constraints"},
                )
                if retry_requirements is not None:
                    final_requirements, _ = self._normalize_requirements_payload(
                        retry_requirements, stage="final_retry"
                    )
            write_json(
                raw_dir / "turnr4_final_requirements_normalized.json", final_requirements
            )
            self._write_requirements_warnings(artifacts_dir / "warnings.json")
            write_json(
                raw_dir / "turnr4_final_requirements_extracted.json", final_requirements
            )
            changelog_raw_context: Dict | None = None
            try:
                changelog = self._extract_marked_json(
                    response.raw_text,
                    "CHANGELOG_JSON:",
                    {"splits", "replacements", "added", "removed"},
                )
                changelog_raw_context = changelog
                write_json(raw_dir / "turnr4_changelog_extracted.json", changelog)
            except ValueError:
                retry_prompt = read_text(
                    self.prompts_dir / "requirements_apply_retry_changelog_only.md"
                )
                retry_payload = {"final_requirements": final_requirements}
                retry_full_prompt = (
                    f"{retry_prompt}\n\nINPUT:\n{json.dumps(retry_payload)}\n"
                )
                retry_prompt_path = raw_dir / "turnr4_apply_changelog_retry_prompt.txt"
                write_text(retry_prompt_path, retry_full_prompt)
                retry_response = adapter.complete(retry_full_prompt)
                retry_raw_path = raw_dir / "turnr4_apply_changelog_retry_raw.txt"
                write_text(retry_raw_path, retry_response.raw_text)
                self._write_usage(
                    raw_dir / "turnr4_apply_changelog_retry_usage.json", retry_response
                )
                try:
                    changelog = self._extract_marked_json(
                        retry_response.raw_text,
                        "CHANGELOG_JSON:",
                        {"splits", "replacements", "added", "removed"},
                    )
                    changelog_raw_context = changelog
                    write_json(raw_dir / "turnr4_changelog_extracted.json", changelog)
                except ValueError as exc:
                    changelog = None
                    changelog_raw_context = {"raw_response": retry_response.raw_text}

            requirements_schema = self._load_schema("normalized_requirements.schema.json")
            try:
                validate(instance=final_requirements, schema=requirements_schema)
            except Exception as exc:
                snippet = json.dumps(final_requirements)[:300]
                raise RuntimeError(
                    "FINAL_REQUIREMENTS_JSON failed validation after repairs. "
                    "See artifacts/repairs_warnings.json and raw turnr4 files. "
                    f"Snippet: {snippet}"
                ) from exc

            final_requirements, id_map = self._normalize_requirements(final_requirements)
            review = self._normalize_review(review, id_map)
            raw_changelog = changelog_raw_context if changelog_raw_context is not None else changelog
            changelog, warnings = self._normalize_changelog(changelog, id_map)
            if changelog is None:
                write_json(artifacts_dir / "changelog_raw.json", raw_changelog)
                write_json(
                    artifacts_dir / "changelog_warnings.json",
                    {"warnings": warnings},
                )
                return final_requirements, {
                    "splits": [],
                    "replacements": [],
                    "added": [],
                    "removed": [],
                    "warnings": warnings,
                }, repairs_applied
            changelog = self._recompute_added(changelog, final_requirements, review)

            failures = self._run_gates(final_requirements, review, changelog, limits)
            if not failures:
                write_normalized_if_repaired(final_requirements, changelog)
                return final_requirements, changelog, repairs_applied

            fix_prompt = read_text(
                self.prompts_dir / "requirements_apply_retry_fix_gates.md"
            )
            fix_prompt = self._render_prompt(fix_prompt, limits)
            fix_payload = {
                "brief": brief,
                "previous": {
                    "final_requirements": final_requirements,
                    "changelog": changelog,
                },
                "review": review,
                "cross_review": cross_review,
                "failures": failures,
                "limits": self._limits_payload(limits),
            }
            fix_full_prompt = f"{fix_prompt}\n\nINPUT:\n{json.dumps(fix_payload)}\n"
            fix_prompt_path = raw_dir / "turnr4_apply_retry_fix_prompt.txt"
            write_text(fix_prompt_path, fix_full_prompt)
            fix_response = adapter.complete(fix_full_prompt)
            write_text(raw_dir / "turnr4_apply_retry_fix_raw.txt", fix_response.raw_text)
            self._write_usage(raw_dir / "turnr4_apply_retry_fix_usage.json", fix_response)

            repair_requirements = self._extract_marked_json(
                fix_response.raw_text,
                "FINAL_REQUIREMENTS_JSON:",
                {"requirements", "assumptions", "constraints"},
            )
            repair_requirements, repair_warnings = self._repair_requirements_payload(
                repair_requirements
            )
            if repair_warnings:
                self._repair_warnings.extend(repair_warnings)
                write_json(
                    artifacts_dir / "repairs_warnings.json",
                    {"warnings": self._repair_warnings},
                )
            repair_requirements, _ = self._normalize_requirements_payload(
                repair_requirements, stage="final_retry"
            )
            repairs_applied = True
            try:
                repair_changelog = self._extract_marked_json(
                    fix_response.raw_text,
                    "CHANGELOG_JSON:",
                    {"splits", "replacements", "added", "removed"},
                )
                write_json(
                    raw_dir / "turnr4_apply_retry_fix_changelog_extracted.json",
                    repair_changelog,
                )
            except ValueError:
                repair_changelog = {
                    "warnings": ["Missing CHANGELOG_JSON in apply fix output."],
                }
            try:
                validate(instance=repair_requirements, schema=requirements_schema)
            except Exception as exc:
                snippet = json.dumps(repair_requirements)[:300]
                raise RuntimeError(
                    "FINAL_REQUIREMENTS_JSON (repair) failed validation after repairs. "
                    "See artifacts/repairs_warnings.json and raw turnr4 files. "
                    f"Snippet: {snippet}"
                ) from exc
            write_json(
                raw_dir / "turnr4_apply_retry_fix_final_requirements_extracted.json",
                repair_requirements,
            )

            merged = self._merge_requirements(final_requirements, repair_requirements)
            changelog = self._merge_changelog(changelog, repair_changelog)
            final_requirements, id_map = self._normalize_requirements(merged)
            review = self._normalize_review(review, id_map)
            raw_changelog = changelog
            changelog, warnings = self._normalize_changelog(changelog, id_map)
            if changelog is None:
                write_json(artifacts_dir / "changelog_raw.json", raw_changelog)
                write_json(
                    artifacts_dir / "changelog_warnings.json",
                    {"warnings": warnings},
                )
                return final_requirements, {
                    "splits": [],
                    "replacements": [],
                    "added": [],
                    "removed": [],
                    "warnings": warnings,
                }, repairs_applied
            changelog = self._recompute_added(changelog, final_requirements, review)

            failures = self._run_gates(final_requirements, review, changelog, limits)
            if not failures:
                write_normalized_if_repaired(final_requirements, changelog)
                return final_requirements, changelog, repairs_applied

            raise RuntimeError(
                f"Requirements apply step failed gates: {', '.join(failures)}"
            )

        raise RuntimeError(f"Requirements apply step failed gates: {', '.join(failures)}")

    def _extract_marked_json(self, raw_text: str, marker: str, expected_keys: set[str]) -> Dict:
        wrapper_key = marker.rstrip(":")
        wrapper_keys: List[str] | None = None
        try:
            parsed_wrapper = extract_json(raw_text)
        except Exception:
            parsed_wrapper = None
        if isinstance(parsed_wrapper, dict):
            wrapper_keys = sorted(parsed_wrapper.keys())
            candidate = parsed_wrapper.get(wrapper_key)
            if isinstance(candidate, dict) and expected_keys.issubset(candidate.keys()):
                self._record_extraction(f"{wrapper_key}:wrapper")
                return candidate
            if expected_keys.issubset(parsed_wrapper.keys()):
                self._record_extraction(f"{wrapper_key}:single-json")
                return parsed_wrapper

        match = re.search(re.escape(marker), raw_text)
        if match:
            snippet = raw_text[match.end():]
            parsed = self._match_expected_json(snippet, expected_keys)
            if parsed is not None:
                self._record_extraction(f"{wrapper_key}:marked-block")
                return parsed

        parsed = self._match_expected_json(raw_text, expected_keys)
        if parsed is not None:
            self._record_extraction(f"{wrapper_key}:raw-json")
            return parsed

        for obj in self._scan_json_objects(raw_text):
            if self._matches_keys(obj, expected_keys):
                self._record_extraction(f"{wrapper_key}:scan")
                return obj

        snippet = raw_text.strip().replace("\n", " ")
        snippet = (snippet[:300] + "...") if len(snippet) > 300 else snippet
        wrapper_note = ""
        if wrapper_keys is not None:
            wrapper_note = f" Detected wrapper JSON with keys: {wrapper_keys}."
        raise ValueError(
            f"Unable to extract JSON for marker {marker}.{wrapper_note} Snippet: {snippet}"
        )

    def _try_parse_wrapper(self, raw_text: str) -> Dict | None:
        try:
            parsed = extract_json(raw_text)
        except Exception:
            return None
        if isinstance(parsed, dict) and (
            "REVIEW_JSON" in parsed
            or "REQUIREMENTS_JSON" in parsed
            or "BUSINESS_RULES_JSON" in parsed
            or "WORKFLOWS_JSON" in parsed
            or "DOMAIN_MODEL_JSON" in parsed
            or "MVP_SCOPE_JSON" in parsed
            or "FINAL_REQUIREMENTS_JSON" in parsed
        ):
            return parsed
        return None

    def _match_expected_json(self, text: str, expected_keys: set[str]) -> Dict | None:
        try:
            parsed = extract_json(text)
        except Exception:
            return None
        if self._matches_keys(parsed, expected_keys):
            return parsed
        return None

    def _scan_json_objects(self, raw_text: str) -> List[Dict]:
        decoder = json.JSONDecoder()
        results: List[Dict] = []
        idx = 0
        while idx < len(raw_text) and len(results) < 2:
            start = self._find_next_json_start(raw_text, idx)
            if start == -1:
                break
            try:
                parsed, end = decoder.raw_decode(raw_text[start:])
                if isinstance(parsed, dict):
                    results.append(parsed)
                idx = start + end
            except json.JSONDecodeError:
                idx = start + 1
        return results

    def _matches_keys(self, obj: object, expected_keys: set[str]) -> bool:
        return isinstance(obj, dict) and expected_keys.issubset(set(obj.keys()))

    def _find_next_json_start(self, text: str, start: int) -> int:
        for idx in range(start, len(text)):
            if text[idx] in "{[":
                return idx
        return -1

    def _run_gates(
        self,
        requirements: Dict,
        review: Dict,
        changelog: Dict,
        limits: RequirementsLimits,
    ) -> List[str]:
        failures: List[str] = []

        req_items = requirements.get("requirements", [])
        req_count = len(req_items)
        if req_count < limits.req_min:
            failures.append(
                f"Gate A: requirements count {req_count} < {limits.req_min}"
            )
        if limits.req_max is not None and req_count > limits.req_max:
            failures.append(
                f"Gate A: requirements count {req_count} > {limits.req_max}"
            )

        if len(requirements.get("assumptions", [])) < limits.assumptions_min:
            failures.append("Gate B: assumptions below minimum")
        if len(requirements.get("constraints", [])) < limits.constraints_min:
            failures.append("Gate B: constraints below minimum")

        rejected_ids = set(review.get("rejected", []))
        accepted_ids = set(review.get("accepted", []))
        added_ids = set(changelog.get("added", []))
        split_targets = {
            item_id
            for split in changelog.get("splits", [])
            for item_id in split.get("into", []) or split.get("to", [])
        }
        added_ids |= split_targets
        final_ids = {item.get("id") for item in req_items}

        if rejected_ids and rejected_ids.intersection(final_ids):
            failures.append("Gate C: rejected requirement IDs present in final")

        if accepted_ids:
            missing_from_acceptance = {
                req_id
                for req_id in final_ids
                if req_id not in accepted_ids and req_id not in added_ids
            }
            if missing_from_acceptance:
                failures.append("Gate C: final requirements include IDs not accepted or added")

        issues = " ".join(review.get("issues", [])).lower()
        needs_split = any(word in issues for word in ["split", "epic"])
        if needs_split:
            splits = changelog.get("splits", []) if isinstance(changelog, dict) else []
            if not splits:
                failures.append("Gate D: review mentions split/epic but changelog has no splits")

        return failures

    def _is_requested(self, artifact: str, limits: RequirementsLimits) -> bool:
        requested = [item.lower() for item in limits.requested_artifacts]
        if not requested:
            return True
        return artifact.lower() in requested

    def _extract_final_artifacts(
        self,
        raw_text: str,
        limits: RequirementsLimits,
        raw_dir: Path,
        suffix: str = "",
    ) -> Dict[str, Dict]:
        artifacts: Dict[str, Dict] = {}
        requested = [
            ("business_rules", "FINAL_BUSINESS_RULES_JSON:", {"rules"}),
            ("workflows", "FINAL_WORKFLOWS_JSON:", {"workflows"}),
            ("domain_model", "FINAL_DOMAIN_MODEL_JSON:", {"entities", "relationships"}),
            ("mvp_scope", "FINAL_MVP_SCOPE_JSON:", {"in_scope", "out_of_scope"}),
        ]
        for key, marker, expected in requested:
            if not self._is_requested(key, limits):
                continue
            try:
                payload = self._extract_marked_json(raw_text, marker, expected)
            except ValueError as exc:
                artifacts[key] = {}
                write_json(
                    raw_dir / f"turnr4_{key}{suffix}_extraction_warning.json",
                    {"warning": str(exc)},
                )
                continue
            artifacts[key] = payload
            write_json(raw_dir / f"turnr4_{key}{suffix}_extracted.json", payload)
        return artifacts

    def _merge_final_artifacts(
        self, base: Dict[str, Dict], repair: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        merged = dict(base)
        merged.update({key: value for key, value in repair.items() if value})
        return merged

    def _validate_and_write_final_artifacts(
        self, artifacts: Dict[str, Dict], artifacts_dir: Path
    ) -> None:
        schema_map = {
            "business_rules": ("business_rules.schema.json", self._write_business_rules_markdown),
            "workflows": ("workflows.schema.json", self._write_workflows_markdown),
            "domain_model": ("domain_model.schema.json", self._write_domain_model_markdown),
            "mvp_scope": ("mvp_scope.schema.json", self._write_mvp_scope_markdown),
        }
        for key, payload in artifacts.items():
            if not payload:
                raise RuntimeError(f"Missing FINAL_{key.upper()}_JSON in apply output.")
            schema_name, writer = schema_map[key]
            self._validate_artifact(payload, schema_name, key)
            write_json(artifacts_dir / f"{key}.json", payload)
            writer(artifacts_dir / f"{key}.md", payload)

    def _validate_artifact(
        self, payload: Dict, schema_name: str, label: str, repair_note: str | None = None
    ) -> None:
        schema = self._load_schema(schema_name)
        try:
            validate(instance=payload, schema=schema)
        except Exception as exc:
            snippet = json.dumps(payload)[:300]
            note = f" Repair note: {repair_note}" if repair_note else ""
            raise RuntimeError(
                f"{label} failed validation.{note} Snippet: {snippet}"
            ) from exc

    def _write_business_rules_markdown(self, path: Path, payload: Dict) -> None:
        lines = ["# Business Rules", ""]
        for rule in payload.get("rules", []):
            rationale = rule.get("rationale", "")
            if rationale:
                lines.append(f"- {rule.get('id')}: {rule.get('text')} (Rationale: {rationale})")
            else:
                lines.append(f"- {rule.get('id')}: {rule.get('text')}")
        write_text(path, "\n".join(lines) + "\n")

    def _normalize_business_rules(self, payload: Dict) -> Dict:
        if not isinstance(payload, dict):
            return {"rules": []}
        rules = payload.get("rules", [])
        if not isinstance(rules, list):
            return {"rules": []}
        normalized: List[Dict[str, str]] = []
        for item in rules:
            if not isinstance(item, dict):
                continue
            rule_id = item.get("id")
            text = item.get("text")
            rationale = item.get("rationale")
            if not all(isinstance(value, str) for value in [rule_id, text, rationale]):
                continue
            normalized.append({"id": rule_id, "text": text, "rationale": rationale})
        return {"rules": normalized}

    def _normalize_str_list(self, items: object, field_name: str, stage: str) -> List[str]:
        normalized: List[str] = []
        if not isinstance(items, list):
            self._requirements_warnings.append(
                {
                    "stage": stage,
                    "field": field_name,
                    "note": "Expected list; replaced with empty list.",
                    "original": items,
                }
            )
            return normalized

        for idx, item in enumerate(items):
            if isinstance(item, str):
                normalized.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("normalized_text")
                if isinstance(text, str):
                    item_id = item.get("id")
                    if isinstance(item_id, str):
                        normalized.append(f"{item_id}: {text}")
                    else:
                        normalized.append(text)
                    self._requirements_warnings.append(
                        {
                            "stage": stage,
                            "field": field_name,
                            "index": idx,
                            "note": "Converted object to text.",
                            "original": item,
                        }
                    )
                    self._list_repair_counts[field_name] += 1
                    continue
                if all(
                    key in item for key in ["id", "text", "category", "priority"]
                ) and isinstance(item.get("text"), str):
                    normalized.append(item.get("text"))
                    self._requirements_warnings.append(
                        {
                            "stage": stage,
                            "field": field_name,
                            "index": idx,
                            "note": "Converted NFR-style object to text.",
                            "original": item,
                        }
                    )
                    self._list_repair_counts[field_name] += 1
                    continue
                serialized = json.dumps(item, ensure_ascii=False)
                normalized.append(serialized)
                self._requirements_warnings.append(
                    {
                        "stage": stage,
                        "field": field_name,
                        "index": idx,
                        "note": "Serialized object to text.",
                        "original": item,
                    }
                )
                self._list_repair_counts[field_name] += 1
                continue
            if item is None or item == "":
                self._requirements_warnings.append(
                    {
                        "stage": stage,
                        "field": field_name,
                        "index": idx,
                        "note": "Dropped empty item.",
                        "original": item,
                    }
                )
                self._list_repair_counts[field_name] += 1
                continue
            normalized.append(str(item))
            self._requirements_warnings.append(
                {
                    "stage": stage,
                    "field": field_name,
                    "index": idx,
                    "note": "Coerced non-string item to text.",
                    "original": item,
                }
            )
            self._list_repair_counts[field_name] += 1
        return normalized

    def _normalize_requirements_payload(
        self, payload: Dict, stage: str
    ) -> tuple[Dict, int]:
        warnings: List[Dict] = []
        string_count = 0
        if not isinstance(payload, dict):
            self._requirements_warnings.append(
                {"stage": stage, "note": "Payload was not an object.", "original": payload}
            )
            return payload, string_count

        items = payload.get("requirements", [])
        if not isinstance(items, list):
            self._requirements_warnings.append(
                {"stage": stage, "note": "Requirements list was not an array.", "original": items}
            )
            payload["requirements"] = []
            return payload, string_count

        assumptions = self._normalize_str_list(
            payload.get("assumptions", []), "assumptions", stage
        )
        constraints = self._normalize_str_list(
            payload.get("constraints", []), "constraints", stage
        )
        payload["assumptions"] = assumptions
        payload["constraints"] = constraints

        existing_ids = {
            item.get("id")
            for item in items
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        auto_index = 1

        def next_auto_id() -> str:
            nonlocal auto_index
            while True:
                candidate = f"REQ-AUTO-{auto_index}"
                auto_index += 1
                if candidate not in existing_ids:
                    existing_ids.add(candidate)
                    return candidate

        normalized_items: List[Dict] = []
        for idx, item in enumerate(items):
            if item is None or item == "":
                warnings.append(
                    {
                        "stage": stage,
                        "index": idx,
                        "note": "Dropped empty requirement item.",
                        "original": item,
                    }
                )
                continue

            if isinstance(item, str):
                string_count += 1
                req_id = next_auto_id()
                normalized_items.append(
                    {"id": req_id, "text": item, "priority": "should"}
                )
                warnings.append(
                    {
                        "stage": stage,
                        "index": idx,
                        "note": "Converted string requirement to object.",
                        "original": item,
                        "repaired_id": req_id,
                    }
                )
                self._list_repair_counts["requirements"] += 1
                continue

            if isinstance(item, dict):
                text = item.get("text") or item.get("normalized_text") or item.get("requirement")
                if not isinstance(text, str) or not text.strip():
                    warnings.append(
                        {
                            "stage": stage,
                            "index": idx,
                            "note": "Dropped requirement missing text.",
                            "original": item,
                        }
                    )
                    self._list_repair_counts["requirements"] += 1
                    continue
                priority = item.get("priority")
                if not isinstance(priority, str) or priority.lower() not in {
                    "must",
                    "should",
                    "could",
                }:
                    priority = "should"
                req_id = item.get("id") if isinstance(item.get("id"), str) else next_auto_id()
                if req_id in existing_ids:
                    req_id = next_auto_id()
                normalized_items.append(
                    {"id": req_id, "text": text.strip(), "priority": priority.lower()}
                )
                continue

            warnings.append(
                {
                    "stage": stage,
                    "index": idx,
                    "note": "Dropped unsupported requirement item.",
                    "original": item,
                }
            )
            self._list_repair_counts["requirements"] += 1

        payload["requirements"] = normalized_items
        if warnings:
            self._requirements_warnings.extend(warnings)
        return payload, string_count

    def _retry_requirements_only(
        self,
        adapter: LLMAdapter,
        brief: str,
        limits: RequirementsLimits,
        raw_dir: Path,
        prompt_name: str,
        prefix: str,
        marker: str,
        expected_keys: set[str],
    ) -> Dict | None:
        retry_template = read_text(self.prompts_dir / prompt_name)
        retry_prompt = self._render_prompt(retry_template, limits)
        retry_full_prompt = f"{retry_prompt}\n\nINPUT:\n{brief}\n"
        retry_prompt_path = raw_dir / f"{prefix}_prompt.txt"
        write_text(retry_prompt_path, retry_full_prompt)
        retry_response = adapter.complete(retry_full_prompt)
        write_text(raw_dir / f"{prefix}_raw.txt", retry_response.raw_text)
        self._write_usage(raw_dir / f"{prefix}_usage.json", retry_response)
        try:
            extracted = self._extract_marked_json(
                retry_response.raw_text, marker, expected_keys
            )
            write_json(raw_dir / f"{prefix}_extracted.json", extracted)
            return extracted
        except ValueError as exc:
            self._requirements_warnings.append(
                {"stage": prefix, "note": "Retry extraction failed.", "error": str(exc)}
            )
            return None

    def _write_requirements_warnings(self, path: Path) -> None:
        if self._requirements_warnings:
            write_json(path, {"warnings": self._requirements_warnings})

    def _delta_retry_artifact(
        self,
        adapter: LLMAdapter,
        prompt_name: str,
        label: str,
        expected_keys: set[str],
        schema_name: str,
        raw_dir: Path,
        artifacts_dir: Path,
        payload: Dict,
        retry_key: str,
    ) -> Dict:
        self._delta_retry_counts[retry_key] = self._delta_retry_counts.get(retry_key, 0) + 1
        prompt = read_text(self.prompts_dir / prompt_name)
        full_prompt = f"{prompt}\n\nINPUT:\n{json.dumps(payload)}\n"
        retry_prefix = f"turn_apply_delta_{retry_key}"
        write_text(raw_dir / f"{retry_prefix}_prompt.txt", full_prompt)
        response = adapter.complete(full_prompt)
        write_text(raw_dir / f"{retry_prefix}_raw.txt", response.raw_text)
        self._write_usage(raw_dir / f"{retry_prefix}_usage.json", response)
        try:
            corrected = self._extract_marked_json(
                response.raw_text, f"{label}:", expected_keys
            )
            write_json(raw_dir / f"{retry_prefix}_extracted.json", corrected)
            self._validate_artifact(corrected, schema_name, label)
            return corrected
        except Exception as exc:
            self._section_warnings.setdefault(retry_key, []).append(str(exc))
            write_json(
                artifacts_dir / f"{retry_key}_warnings.json",
                {"warnings": self._section_warnings[retry_key]},
            )
            return payload

    def _load_cached_artifact(self, path: Path, fallback: Dict) -> Dict:
        if path.exists():
            try:
                cached = json.loads(read_text(path))
                if isinstance(cached, dict):
                    return cached
            except Exception:
                return fallback
        return fallback

    def _require_section(
        self,
        section_name: str,
        label: str,
        schema_name: str,
        retry_prompt_path: str,
        default_value: Dict,
        expected_keys: set[str],
        brief: str,
        limits: RequirementsLimits,
        adapter: LLMAdapter,
        raw_dir: Path,
        artifacts_dir: Path,
    ) -> Dict:
        warnings = self._section_warnings.setdefault(section_name, [])
        warnings.append(f"Missing {label} in lead output.")
        write_json(
            artifacts_dir / f"{section_name}_warnings.json",
            {"warnings": warnings},
        )

        retry_template = read_text(self.prompts_dir / retry_prompt_path)
        retry_prompt = self._render_prompt(retry_template, limits)
        retry_full_prompt = f"{retry_prompt}\n\nINPUT:\n{brief}\n"
        retry_prompt_path = raw_dir / f"turnr1_{section_name}_retry_prompt.txt"
        write_text(retry_prompt_path, retry_full_prompt)
        retry_response = adapter.complete(retry_full_prompt)
        write_text(
            raw_dir / f"turnr1_{section_name}_retry_raw.txt", retry_response.raw_text
        )
        self._write_usage(
            raw_dir / f"turnr1_{section_name}_retry_usage.json", retry_response
        )

        try:
            extracted = self._extract_marked_json(
                retry_response.raw_text,
                f"{label}:",
                expected_keys,
            )
            write_json(
                raw_dir / f"turnr1_{section_name}_retry_extracted.json", extracted
            )
            normalized = (
                self._normalize_business_rules(extracted)
                if section_name == "business_rules"
                else extracted
            )
            self._validate_artifact(normalized, schema_name, label)
            return normalized
        except ValueError as exc:
            warnings.append(str(exc))
        except Exception as exc:
            warnings.append(f"Validation failed: {exc}")

        write_json(
            artifacts_dir / f"{section_name}_warnings.json",
            {"warnings": warnings},
        )
        return default_value

    def _write_workflows_markdown(self, path: Path, payload: Dict) -> None:
        lines = ["# Workflows", ""]
        for workflow in payload.get("workflows", []):
            lines.append(f"## {workflow.get('name')} ({workflow.get('id')})")
            states = workflow.get("states", [])
            if states:
                lines.append(f"- States: {', '.join(states)}")
            transitions = workflow.get("transitions", [])
            if transitions:
                lines.append("- Transitions:")
                for transition in transitions:
                    lines.append(
                        f"  - {transition.get('from')} -> {transition.get('to')}: "
                        f"{transition.get('trigger')}"
                    )
            lines.append("")
        write_text(path, "\n".join(lines).rstrip() + "\n")

    def _normalize_workflows(self, payload: Dict) -> Dict:
        if not isinstance(payload, dict):
            return {"workflows": []}
        workflows = payload.get("workflows", [])
        if not isinstance(workflows, list):
            return {"workflows": []}
        normalized: List[Dict] = []
        for workflow in workflows:
            if not isinstance(workflow, dict):
                continue
            workflow_id = workflow.get("id")
            if not isinstance(workflow_id, str):
                continue
            name = workflow.get("name")
            if not isinstance(name, str) or not name.strip():
                name = f"Workflow {workflow_id}"
            states = workflow.get("states", [])
            if not isinstance(states, list):
                states = []
            states = [state for state in states if isinstance(state, str)]
            transitions = workflow.get("transitions", [])
            if not isinstance(transitions, list):
                transitions = []
            normalized_transitions: List[Dict[str, str]] = []
            for transition in transitions:
                if not isinstance(transition, dict):
                    continue
                from_state = transition.get("from")
                to_state = transition.get("to")
                trigger = transition.get("trigger")
                if not isinstance(trigger, str):
                    guard = transition.get("guard")
                    if isinstance(guard, str):
                        trigger = guard
                if (
                    isinstance(from_state, str)
                    and isinstance(to_state, str)
                    and isinstance(trigger, str)
                ):
                    normalized_transitions.append(
                        {"from": from_state, "to": to_state, "trigger": trigger}
                    )
            normalized.append(
                {
                    "id": workflow_id,
                    "name": name,
                    "states": states,
                    "transitions": normalized_transitions,
                }
            )
        return {"workflows": normalized}

    def _repair_workflows(self, payload: Dict) -> tuple[Dict, List[str]]:
        warnings: List[str] = []
        if not isinstance(payload, dict):
            return {"workflows": []}, ["Payload was not an object."]
        workflows = payload.get("workflows", [])
        if not isinstance(workflows, list):
            return {"workflows": []}, ["Workflows was not a list."]
        repaired: List[Dict] = []
        for workflow in workflows:
            if not isinstance(workflow, dict):
                continue
            workflow_id = workflow.get("id")
            if not isinstance(workflow_id, str):
                continue
            name = workflow.get("name")
            if not isinstance(name, str) or not name.strip():
                name = f"Workflow {workflow_id}"
                warnings.append(f"Filled missing name for workflow {workflow_id}.")
            states = workflow.get("states", [])
            if not isinstance(states, list):
                states = []
            states = [state for state in states if isinstance(state, str)]
            transitions = workflow.get("transitions", [])
            if not isinstance(transitions, list):
                transitions = []
            repaired_transitions: List[Dict[str, str]] = []
            for transition in transitions:
                if not isinstance(transition, dict):
                    continue
                from_state = transition.get("from")
                to_state = transition.get("to")
                trigger = transition.get("trigger")
                if not isinstance(trigger, str):
                    guard = transition.get("guard")
                    if isinstance(guard, str):
                        trigger = guard
                        warnings.append("Renamed guard to trigger in transition.")
                if (
                    isinstance(from_state, str)
                    and isinstance(to_state, str)
                    and isinstance(trigger, str)
                ):
                    repaired_transitions.append(
                        {"from": from_state, "to": to_state, "trigger": trigger}
                    )
            repaired.append(
                {
                    "id": workflow_id,
                    "name": name,
                    "states": states,
                    "transitions": repaired_transitions,
                }
            )
        return {"workflows": repaired}, warnings

    def _repair_domain_model(self, payload: Dict) -> tuple[Dict, List[str]]:
        warnings: List[str] = []
        if not isinstance(payload, dict):
            return {"entities": [], "relationships": []}, ["Payload was not an object."]
        entities = payload.get("entities", [])
        relationships = payload.get("relationships", [])
        entities = entities if isinstance(entities, list) else []
        relationships = relationships if isinstance(relationships, list) else []
        repaired_relationships: List[Dict] = []
        for relation in relationships:
            if not isinstance(relation, dict):
                continue
            from_entity = relation.get("from")
            to_entity = relation.get("to")
            rel_type = relation.get("type")
            description = relation.get("description")
            if not isinstance(description, str) and all(
                isinstance(value, str) for value in [from_entity, rel_type, to_entity]
            ):
                description = f"{from_entity} {rel_type} {to_entity} relationship."
                warnings.append(
                    f"Filled missing description for relationship {from_entity}->{to_entity}."
                )
            if all(isinstance(value, str) for value in [from_entity, to_entity, rel_type, description]):
                repaired_relationships.append(
                    {
                        "from": from_entity,
                        "to": to_entity,
                        "type": rel_type,
                        "description": description,
                    }
                )
        repaired_entities: List[Dict] = []
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            name = entity.get("name")
            description = entity.get("description")
            attributes = entity.get("attributes", [])
            if not isinstance(attributes, list):
                attributes = []
            repaired_attributes: List[Dict] = []
            for attr in attributes:
                if not isinstance(attr, dict):
                    continue
                attr_name = attr.get("name")
                attr_type = attr.get("type")
                attr_desc = attr.get("description")
                if all(isinstance(value, str) for value in [attr_name, attr_type, attr_desc]):
                    repaired_attributes.append(
                        {"name": attr_name, "type": attr_type, "description": attr_desc}
                    )
            if all(isinstance(value, str) for value in [name, description]):
                repaired_entities.append(
                    {"name": name, "description": description, "attributes": repaired_attributes}
                )
        repaired = {"entities": repaired_entities, "relationships": repaired_relationships}
        return repaired, warnings

    def _repair_mvp_scope(self, payload: Dict) -> tuple[Dict, List[str]]:
        warnings: List[str] = []
        if not isinstance(payload, dict):
            return {"in_scope": [], "out_of_scope": [], "milestones": []}, [
                "Payload was not an object."
            ]
        in_scope = payload.get("in_scope", [])
        out_of_scope = payload.get("out_of_scope", [])
        milestones = payload.get("milestones", [])
        if isinstance(in_scope, str):
            in_scope = [in_scope]
        if isinstance(out_of_scope, str):
            out_of_scope = [out_of_scope]
        if isinstance(milestones, str):
            milestones = [milestones]
        in_scope = [item for item in in_scope if isinstance(item, str)] if isinstance(in_scope, list) else []
        out_of_scope = (
            [item for item in out_of_scope if isinstance(item, str)]
            if isinstance(out_of_scope, list)
            else []
        )
        repaired_milestones: List[Dict[str, str]] = []
        if isinstance(milestones, list):
            for item in milestones:
                if isinstance(item, str):
                    repaired_milestones.append(
                        {"name": item, "description": f"Milestone: {item}"}
                    )
                    warnings.append("Converted milestone string to object.")
                    continue
                if isinstance(item, dict):
                    name = item.get("name")
                    description = item.get("description")
                    if isinstance(name, str):
                        if not isinstance(description, str):
                            description = f"Milestone: {name}"
                            warnings.append(f"Filled missing milestone description for {name}.")
                        repaired_milestones.append(
                            {"name": name, "description": description}
                        )
        repaired = {
            "in_scope": in_scope,
            "out_of_scope": out_of_scope,
            "milestones": repaired_milestones,
        }
        return repaired, warnings

    def _write_domain_model_markdown(self, path: Path, payload: Dict) -> None:
        lines = ["# Domain Model", ""]
        for entity in payload.get("entities", []):
            lines.append(f"## {entity.get('name')}")
            lines.append(entity.get("description", ""))
            attributes = entity.get("attributes", [])
            if attributes:
                lines.append("- Attributes:")
                for attr in attributes:
                    lines.append(
                        f"  - {attr.get('name')} ({attr.get('type')}): "
                        f"{attr.get('description')}"
                    )
            lines.append("")
        relationships = payload.get("relationships", [])
        if relationships:
            lines.append("## Relationships")
            for relation in relationships:
                lines.append(
                    f"- {relation.get('from')} -> {relation.get('to')}: "
                    f"{relation.get('type')} ({relation.get('description')})"
                )
        write_text(path, "\n".join(lines).rstrip() + "\n")

    def _write_mvp_scope_markdown(self, path: Path, payload: Dict) -> None:
        lines = ["# MVP Scope", "", "## In Scope"]
        for item in payload.get("in_scope", []):
            lines.append(f"- {item}")
        lines.append("")
        lines.append("## Out of Scope")
        for item in payload.get("out_of_scope", []):
            lines.append(f"- {item}")
        milestones = payload.get("milestones", [])
        if milestones:
            lines.append("")
            lines.append("## Milestones")
            for milestone in milestones:
                lines.append(f"- {milestone.get('name')}: {milestone.get('description')}")
        write_text(path, "\n".join(lines).rstrip() + "\n")

    def _must_requirements(self, requirements: Dict) -> List[Dict]:
        return [
            item
            for item in requirements.get("requirements", [])
            if str(item.get("priority", "")).lower() == "must"
        ]

    def _run_acceptance_criteria(
        self,
        chatgpt: LLMAdapter,
        gemini: LLMAdapter,
        final_requirements: Dict,
        raw_dir: Path,
        artifacts_dir: Path,
    ) -> Dict | None:
        must_requirements = self._must_requirements(final_requirements)
        if not must_requirements:
            self._acceptance_warnings.append("No MUST requirements for acceptance criteria.")
            return None

        acceptance_prompt = read_text(self.prompts_dir / "acceptance_chatgpt.md")
        acceptance_payload = {"must_requirements": must_requirements}
        acceptance_full_prompt = (
            f"{acceptance_prompt}\n\nINPUT:\n{json.dumps(acceptance_payload)}\n"
        )
        acceptance_prompt_path = raw_dir / "turnr6_acceptance_prompt.txt"
        write_text(acceptance_prompt_path, acceptance_full_prompt)
        acceptance_response = chatgpt.complete(acceptance_full_prompt)
        write_text(raw_dir / "turnr6_acceptance_raw.txt", acceptance_response.raw_text)
        self._write_usage(raw_dir / "turnr6_acceptance_usage.json", acceptance_response)

        try:
            acceptance_json = self._extract_marked_json(
                acceptance_response.raw_text,
                "ACCEPTANCE_CRITERIA_JSON:",
                {"criteria"},
            )
        except ValueError as exc:
            self._acceptance_warnings.append(str(exc))
            write_json(
                artifacts_dir / "acceptance_warnings.json",
                {"warnings": self._acceptance_warnings},
            )
            return None

        write_json(raw_dir / "turnr6_acceptance_extracted.json", acceptance_json)

        cross_prompt = read_text(self.prompts_dir / "acceptance_gemini_cross_review.md")
        cross_payload = {
            "must_requirements": must_requirements,
            "acceptance_criteria": acceptance_json,
        }
        cross_full_prompt = f"{cross_prompt}\n\nINPUT:\n{json.dumps(cross_payload)}\n"
        cross_prompt_path = raw_dir / "turnr7_acceptance_cross_prompt.txt"
        write_text(cross_prompt_path, cross_full_prompt)
        cross_response = gemini.complete(cross_full_prompt)
        write_text(raw_dir / "turnr7_acceptance_cross_raw.txt", cross_response.raw_text)
        self._write_usage(raw_dir / "turnr7_acceptance_cross_usage.json", cross_response)

        cross_review = extract_json(cross_response.raw_text)
        if not isinstance(cross_review, dict):
            cross_review = {"issues": ["Cross review output was not JSON."]}
        write_json(raw_dir / "turnr7_acceptance_cross_review.json", cross_review)
        write_json(artifacts_dir / "acceptance_cross_review.json", cross_review)

        finalize_prompt = read_text(self.prompts_dir / "acceptance_chatgpt_finalize.md")
        finalize_payload = {
            "must_requirements": must_requirements,
            "draft_acceptance": acceptance_json,
            "cross_review": cross_review,
        }
        finalize_full_prompt = f"{finalize_prompt}\n\nINPUT:\n{json.dumps(finalize_payload)}\n"
        finalize_prompt_path = raw_dir / "turnr8_acceptance_finalize_prompt.txt"
        write_text(finalize_prompt_path, finalize_full_prompt)
        finalize_response = chatgpt.complete(finalize_full_prompt)
        write_text(raw_dir / "turnr8_acceptance_finalize_raw.txt", finalize_response.raw_text)
        self._write_usage(raw_dir / "turnr8_acceptance_finalize_usage.json", finalize_response)

        try:
            final_acceptance = self._extract_marked_json(
                finalize_response.raw_text,
                "ACCEPTANCE_CRITERIA_JSON:",
                {"criteria"},
            )
        except ValueError as exc:
            self._acceptance_warnings.append(str(exc))
            write_json(
                artifacts_dir / "acceptance_warnings.json",
                {"warnings": self._acceptance_warnings},
            )
            return None

        schema = self._load_schema("acceptance_criteria.schema.json")
        try:
            validate(instance=final_acceptance, schema=schema)
        except Exception as exc:
            self._acceptance_warnings.append(f"Validation failed: {exc}")
            write_json(
                artifacts_dir / "acceptance_warnings.json",
                {"warnings": self._acceptance_warnings},
            )
            return None

        invalid_ids = self._validate_acceptance_ids(final_acceptance, must_requirements)
        if invalid_ids:
            self._acceptance_warnings.append(
                f"Acceptance criteria contains non-MUST IDs: {', '.join(invalid_ids)}"
            )
            write_json(
                artifacts_dir / "acceptance_warnings.json",
                {"warnings": self._acceptance_warnings},
            )
            return None

        write_json(artifacts_dir / "acceptance_criteria.json", final_acceptance)
        self._write_acceptance_markdown(
            artifacts_dir / "acceptance_criteria.md", final_acceptance
        )
        return final_acceptance

    def _validate_acceptance_ids(
        self, acceptance: Dict, must_requirements: List[Dict]
    ) -> List[str]:
        must_ids = {item.get("id") for item in must_requirements if item.get("id")}
        invalid = []
        for entry in acceptance.get("criteria", []):
            req_id = entry.get("requirement_id")
            if req_id not in must_ids:
                invalid.append(str(req_id))
        return invalid

    def _write_acceptance_markdown(self, path: Path, payload: Dict) -> None:
        lines = ["# Acceptance Criteria", ""]
        for entry in payload.get("criteria", []):
            lines.append(f"## {entry.get('requirement_id')}")
            for item in entry.get("criteria", []):
                lines.append(f"- {item}")
            lines.append("")
        write_text(path, "\n".join(lines).rstrip() + "\n")

    def _run_apply_stage_b(
        self,
        adapter: LLMAdapter,
        brief: str,
        final_requirements: Dict,
        limits: RequirementsLimits,
        raw_dir: Path,
        artifacts_dir: Path,
    ) -> Dict[str, Dict]:
        if not self._is_requested("business_rules", limits) and not self._is_requested(
            "workflows", limits
        ):
            return {}
        prompt = self._render_prompt(
            read_text(self.prompts_dir / "requirements_apply_stage_b.md"), limits
        )
        payload = {"brief": brief, "requirements": final_requirements}
        full_prompt = f"{prompt}\n\nINPUT:\n{json.dumps(payload)}\n"
        write_text(raw_dir / "turn_apply_stage_b_prompt.txt", full_prompt)
        response = adapter.complete(full_prompt)
        write_text(raw_dir / "turn_apply_stage_b_raw.txt", response.raw_text)
        self._write_usage(raw_dir / "turn_apply_stage_b_usage.json", response)

        results: Dict[str, Dict] = {}
        try:
            business_rules = self._extract_marked_json(
                response.raw_text,
                "FINAL_BUSINESS_RULES_JSON:",
                {"rules"},
            )
            write_json(raw_dir / "turn_apply_stage_b_business_rules_extracted.json", business_rules)
            business_rules = self._normalize_business_rules(business_rules)
            self._validate_artifact(
                business_rules, "business_rules.schema.json", "FINAL_BUSINESS_RULES_JSON"
            )
            results["business_rules"] = business_rules
            write_json(artifacts_dir / "business_rules.json", business_rules)
            self._write_business_rules_markdown(
                artifacts_dir / "business_rules.md", business_rules
            )
        except Exception as exc:
            self._section_warnings.setdefault("business_rules", []).append(str(exc))
            write_json(
                artifacts_dir / "business_rules_warnings.json",
                {"warnings": self._section_warnings["business_rules"]},
            )
            results["business_rules"] = {"rules": []}

        try:
            workflows = self._extract_marked_json(
                response.raw_text,
                "FINAL_WORKFLOWS_JSON:",
                {"workflows"},
            )
            write_json(raw_dir / "turn_apply_stage_b_workflows_extracted.json", workflows)
            write_json(raw_dir / "workflows_raw.json", workflows)
            repaired, warnings = self._repair_workflows(workflows)
            if warnings:
                self._artifact_repair_counts["workflows"] = (
                    self._artifact_repair_counts.get("workflows", 0) + len(warnings)
                )
                write_json(
                    artifacts_dir / "repairs_workflows.json",
                    {"warnings": warnings, "repaired": repaired},
                )
            workflows = self._normalize_workflows(repaired)
            write_json(artifacts_dir / "workflows_normalized.json", workflows)
            try:
                self._validate_artifact(
                    workflows,
                    "workflows.schema.json",
                    "FINAL_WORKFLOWS_JSON",
                    repair_note="Applied workflow repairs before validation.",
                )
                self._artifact_validation["workflows"] = "valid"
            except Exception as exc:
                workflows = self._delta_retry_artifact(
                    adapter=adapter,
                    prompt_name="requirements_apply_retry_workflows_only.md",
                    label="FINAL_WORKFLOWS_JSON",
                    expected_keys={"workflows"},
                    schema_name="workflows.schema.json",
                    raw_dir=raw_dir,
                    artifacts_dir=artifacts_dir,
                    payload=workflows,
                    retry_key="workflows",
                )
                self._section_warnings.setdefault("workflows", []).append(str(exc))
            results["workflows"] = workflows
            write_json(artifacts_dir / "workflows.json", workflows)
            self._write_workflows_markdown(artifacts_dir / "workflows.md", workflows)
        except Exception as exc:
            self._section_warnings.setdefault("workflows", []).append(str(exc))
            write_json(
                artifacts_dir / "workflows_warnings.json",
                {"warnings": self._section_warnings["workflows"]},
            )
            cached = self._load_cached_artifact(artifacts_dir / "workflows.json", {"workflows": []})
            if cached.get("workflows"):
                self._section_warnings["workflows"].append(
                    "Reused cached workflows.json due to missing FINAL_WORKFLOWS_JSON."
                )
                write_json(
                    artifacts_dir / "workflows_warnings.json",
                    {"warnings": self._section_warnings["workflows"]},
                )
                results["workflows"] = cached
            else:
                results["workflows"] = {"workflows": []}

        return results

    def _run_apply_stage_c(
        self,
        adapter: LLMAdapter,
        brief: str,
        final_requirements: Dict,
        limits: RequirementsLimits,
        raw_dir: Path,
        artifacts_dir: Path,
    ) -> Dict[str, Dict]:
        if not self._is_requested("domain_model", limits) and not self._is_requested(
            "mvp_scope", limits
        ):
            return {}
        prompt = self._render_prompt(
            read_text(self.prompts_dir / "requirements_apply_stage_c.md"), limits
        )
        payload = {"brief": brief, "requirements": final_requirements}
        full_prompt = f"{prompt}\n\nINPUT:\n{json.dumps(payload)}\n"
        write_text(raw_dir / "turn_apply_stage_c_prompt.txt", full_prompt)
        response = adapter.complete(full_prompt)
        write_text(raw_dir / "turn_apply_stage_c_raw.txt", response.raw_text)
        self._write_usage(raw_dir / "turn_apply_stage_c_usage.json", response)

        results: Dict[str, Dict] = {}
        try:
            domain_model = self._extract_marked_json(
                response.raw_text,
                "FINAL_DOMAIN_MODEL_JSON:",
                {"entities", "relationships"},
            )
            write_json(raw_dir / "turn_apply_stage_c_domain_model_extracted.json", domain_model)
            write_json(raw_dir / "domain_model_raw.json", domain_model)
            repaired, warnings = self._repair_domain_model(domain_model)
            if warnings:
                self._artifact_repair_counts["domain_model"] = (
                    self._artifact_repair_counts.get("domain_model", 0) + len(warnings)
                )
                write_json(
                    artifacts_dir / "repairs_domain_model.json",
                    {"warnings": warnings, "repaired": repaired},
                )
            try:
                self._validate_artifact(
                    repaired,
                    "domain_model.schema.json",
                    "FINAL_DOMAIN_MODEL_JSON",
                    repair_note="Applied domain model repairs before validation.",
                )
                self._artifact_validation["domain_model"] = "valid"
            except Exception as exc:
                repaired = self._delta_retry_artifact(
                    adapter=adapter,
                    prompt_name="requirements_apply_retry_domain_model_only.md",
                    label="FINAL_DOMAIN_MODEL_JSON",
                    expected_keys={"entities", "relationships"},
                    schema_name="domain_model.schema.json",
                    raw_dir=raw_dir,
                    artifacts_dir=artifacts_dir,
                    payload=repaired,
                    retry_key="domain_model",
                )
                self._section_warnings.setdefault("domain_model", []).append(str(exc))
            results["domain_model"] = repaired
            write_json(artifacts_dir / "domain_model.json", repaired)
            self._write_domain_model_markdown(
                artifacts_dir / "domain_model.md", repaired
            )
        except Exception as exc:
            self._section_warnings.setdefault("domain_model", []).append(str(exc))
            write_json(
                artifacts_dir / "domain_model_warnings.json",
                {"warnings": self._section_warnings["domain_model"]},
            )
            cached = self._load_cached_artifact(
                artifacts_dir / "domain_model.json", {"entities": [], "relationships": []}
            )
            if cached.get("entities") or cached.get("relationships"):
                self._section_warnings["domain_model"].append(
                    "Reused cached domain_model.json due to missing FINAL_DOMAIN_MODEL_JSON."
                )
                write_json(
                    artifacts_dir / "domain_model_warnings.json",
                    {"warnings": self._section_warnings["domain_model"]},
                )
                results["domain_model"] = cached
            else:
                results["domain_model"] = {"entities": [], "relationships": []}

        try:
            mvp_scope = self._extract_marked_json(
                response.raw_text,
                "FINAL_MVP_SCOPE_JSON:",
                {"in_scope", "out_of_scope"},
            )
            write_json(raw_dir / "turn_apply_stage_c_mvp_scope_extracted.json", mvp_scope)
            write_json(raw_dir / "mvp_scope_raw.json", mvp_scope)
            repaired, warnings = self._repair_mvp_scope(mvp_scope)
            write_json(artifacts_dir / "mvp_scope_repaired.json", repaired)
            if warnings:
                self._artifact_repair_counts["mvp_scope"] = (
                    self._artifact_repair_counts.get("mvp_scope", 0) + len(warnings)
                )
                write_json(
                    artifacts_dir / "repairs_mvp_scope.json",
                    {"warnings": warnings, "repaired": repaired},
                )
            self._validate_artifact(
                repaired,
                "mvp_scope.schema.json",
                "FINAL_MVP_SCOPE_JSON",
                repair_note="Applied MVP scope repairs before validation.",
            )
            self._artifact_validation["mvp_scope"] = "valid"
            results["mvp_scope"] = repaired
            write_json(artifacts_dir / "mvp_scope.json", repaired)
            self._write_mvp_scope_markdown(artifacts_dir / "mvp_scope.md", repaired)
        except Exception as exc:
            self._section_warnings.setdefault("mvp_scope", []).append(str(exc))
            write_json(
                artifacts_dir / "mvp_scope_warnings.json",
                {"warnings": self._section_warnings["mvp_scope"]},
            )
            cached = self._load_cached_artifact(
                artifacts_dir / "mvp_scope.json",
                {"in_scope": [], "out_of_scope": [], "milestones": []},
            )
            if cached.get("in_scope") or cached.get("out_of_scope"):
                self._section_warnings["mvp_scope"].append(
                    "Reused cached mvp_scope.json due to missing FINAL_MVP_SCOPE_JSON."
                )
                write_json(
                    artifacts_dir / "mvp_scope_warnings.json",
                    {"warnings": self._section_warnings["mvp_scope"]},
                )
                results["mvp_scope"] = cached
            else:
                results["mvp_scope"] = {"in_scope": [], "out_of_scope": [], "milestones": []}

        return results
    def _check_coverage(self, requirements: Dict, limits: RequirementsLimits) -> Dict:
        req_items = requirements.get("requirements", [])
        req_texts = [str(item.get("text", "")).lower() for item in req_items]
        missing_areas: List[str] = []
        min_per_area = limits.min_per_area or 1
        if limits.coverage_areas:
            for area in limits.coverage_areas:
                area_lower = area.lower()
                count = sum(1 for text in req_texts if area_lower in text)
                if count < min_per_area:
                    missing_areas.append(area)

        missing_seeds: List[str] = []
        for seed in limits.seed_requirements:
            seed_text = ""
            if isinstance(seed, str):
                seed_text = seed
            elif isinstance(seed, dict):
                seed_text = str(seed.get("text") or seed.get("id") or "")
            seed_text = seed_text.strip()
            if not seed_text:
                continue
            seed_lower = seed_text.lower()
            if not any(seed_lower in text for text in req_texts):
                missing_seeds.append(seed_text)

        req_count = len(req_items)
        missing_count = max(limits.req_min - req_count, 0)
        needs_retry = bool(missing_areas or missing_seeds or missing_count)
        return {
            "missing_areas": missing_areas,
            "missing_seeds": missing_seeds,
            "missing_count": missing_count,
            "needs_retry": needs_retry,
            "req_count": req_count,
        }

    def _run_coverage_retry(
        self,
        adapter: LLMAdapter,
        brief: str,
        draft_requirements: Dict,
        coverage: Dict,
        limits: RequirementsLimits,
        raw_dir: Path,
    ) -> Dict | None:
        if not coverage.get("needs_retry"):
            return None
        retry_template = read_text(
            self.prompts_dir / "requirements_lead_retry_missing_coverage.md"
        )
        retry_prompt = self._render_prompt(retry_template, limits)
        retry_payload = {
            "brief": brief,
            "existing_requirements": draft_requirements,
            "missing_areas": coverage.get("missing_areas", []),
            "missing_seeds": coverage.get("missing_seeds", []),
            "missing_count": coverage.get("missing_count", 0),
        }
        retry_full_prompt = f"{retry_prompt}\n\nINPUT:\n{json.dumps(retry_payload)}\n"
        retry_prompt_path = raw_dir / "turnr1_coverage_retry_prompt.txt"
        write_text(retry_prompt_path, retry_full_prompt)
        retry_response = adapter.complete(retry_full_prompt)
        retry_raw_path = raw_dir / "turnr1_coverage_retry_raw.txt"
        write_text(retry_raw_path, retry_response.raw_text)
        self._write_usage(raw_dir / "turnr1_coverage_retry_usage.json", retry_response)
        try:
            retries = self._extract_marked_json(
                retry_response.raw_text,
                "REQUIREMENTS_JSON:",
                {"requirements", "assumptions", "constraints"},
            )
            write_json(raw_dir / "turnr1_coverage_retry_requirements.json", retries)
            return retries
        except ValueError as exc:
            write_json(
                raw_dir / "turnr1_coverage_retry_warning.json",
                {"warning": str(exc)},
            )
            return None

    def _normalize_requirements(self, payload: Dict) -> Tuple[Dict, Dict[str, str]]:
        items = payload.get("requirements", [])
        mapping: Dict[str, str] = {}
        normalized_items: List[Dict] = []
        next_id = 1
        for item in items:
            old_id = str(item.get("id", "")).strip()
            new_id = f"REQ-{next_id}"
            next_id += 1
            if old_id:
                mapping[old_id] = new_id
            normalized_items.append(
                {
                    "id": new_id,
                    "text": str(item.get("text", "")).strip(),
                    "priority": str(item.get("priority", "must")),
                }
            )
        return {
            "requirements": normalized_items,
            "assumptions": payload.get("assumptions", []),
            "constraints": payload.get("constraints", []),
        }, mapping

    def _repair_requirements_payload(self, payload: Dict) -> tuple[Dict, List[Dict]]:
        warnings: List[Dict] = []
        if not isinstance(payload, dict):
            warnings.append(
                {
                    "warning": "Requirements payload is not a JSON object.",
                    "original": payload,
                }
            )
            return payload, warnings

        items = payload.get("requirements", [])
        if not isinstance(items, list):
            warnings.append(
                {
                    "warning": "Requirements list is not an array.",
                    "original": items,
                }
            )
            items = []

        existing_ids: set[str] = set()
        next_id = 1

        def allocate_id() -> str:
            nonlocal next_id
            while True:
                candidate = f"REQ-{next_id:03d}"
                next_id += 1
                if candidate not in existing_ids:
                    existing_ids.add(candidate)
                    return candidate

        def infer_priority(text: str) -> str:
            match = re.match(r"\s*(must|should|could|shall)\b", text, re.IGNORECASE)
            if match:
                value = match.group(1).lower()
                return "must" if value == "shall" else value
            return "should"

        def strip_priority_prefix(text: str) -> str:
            return re.sub(r"^\s*(must|should|could|shall)\b[:\-\s]*", "", text, flags=re.IGNORECASE)

        normalized_items: List[Dict] = []
        for index, item in enumerate(items):
            if isinstance(item, str):
                text = strip_priority_prefix(item.strip())
                req_id = allocate_id()
                priority = infer_priority(item)
                warnings.append(
                    {
                        "index": index,
                        "original": item,
                        "repaired_id": req_id,
                        "note": "Converted string requirement to object.",
                    }
                )
                normalized_items.append(
                    {"id": req_id, "text": text, "priority": priority}
                )
                continue

            if isinstance(item, dict):
                repair_notes: List[str] = []
                text = item.get("text") or item.get("normalized_text") or item.get("requirement")
                if text is None:
                    warnings.append(
                        {
                            "index": index,
                            "original": item,
                            "repaired_id": None,
                            "note": "Dropped requirement with no text field.",
                        }
                    )
                    continue
                text_str = str(text).strip()
                if "text" not in item and ("normalized_text" in item or "requirement" in item):
                    repair_notes.append("Normalized text field from alternate key.")
                raw_priority = item.get("priority")
                if isinstance(raw_priority, str):
                    priority = raw_priority.lower()
                else:
                    priority = infer_priority(text_str)
                    repair_notes.append("Inferred priority for requirement.")
                if priority not in {"must", "should", "could"}:
                    priority = infer_priority(text_str)
                    repair_notes.append("Repaired invalid priority value.")

                req_id = item.get("id") if isinstance(item.get("id"), str) else None
                if not req_id:
                    req_id = allocate_id()
                    repair_notes.append("Assigned new id for missing requirement id.")
                elif req_id in existing_ids:
                    new_id = allocate_id()
                    repair_notes.append("Assigned new id for duplicate requirement id.")
                    req_id = new_id
                existing_ids.add(req_id)
                if repair_notes:
                    warnings.append(
                        {
                            "index": index,
                            "original": item,
                            "repaired_id": req_id,
                            "note": "; ".join(repair_notes),
                        }
                    )
                normalized_items.append(
                    {"id": req_id, "text": text_str, "priority": priority}
                )
                continue

            warnings.append(
                {
                    "index": index,
                    "original": item,
                    "repaired_id": None,
                    "note": "Dropped requirement with unsupported type.",
                }
            )

        payload["requirements"] = normalized_items
        return payload, warnings

    def _normalize_review(self, review: Dict, mapping: Dict[str, str]) -> Dict:
        def map_ids(ids: List[str]) -> List[str]:
            return [mapping[item] for item in ids if item in mapping]

        return {
            "accepted": map_ids(review.get("accepted", [])),
            "rejected": map_ids(review.get("rejected", [])),
            "issues": review.get("issues", []),
            "missing": review.get("missing", []),
            "rationale": review.get("rationale", []),
        }

    def normalize_review_json(self, review: Dict) -> Dict:
        def normalize_ids(key: str) -> List[str]:
            normalized: List[str] = []
            values = review.get(key, [])
            if not isinstance(values, list):
                return normalized
            for item in values:
                if isinstance(item, str):
                    normalized.append(item)
                elif isinstance(item, dict):
                    candidate = item.get("id")
                    if isinstance(candidate, str):
                        normalized.append(candidate)
                        self._review_normalization_warnings.append(
                            f"Normalized {key} entry object to id: {candidate}"
                        )
                    else:
                        self._review_normalization_warnings.append(
                            f"Dropped {key} entry without string id: {item}"
                        )
                else:
                    self._review_normalization_warnings.append(
                        f"Dropped {key} entry with unsupported type: {item}"
                    )
            return normalized

        return {
            "accepted": normalize_ids("accepted"),
            "rejected": normalize_ids("rejected"),
            "issues": review.get("issues", []),
            "missing": review.get("missing", []),
            "rationale": review.get("rationale", []),
        }

    def _normalize_changelog(
        self, changelog: Dict | None, mapping: Dict[str, str]
    ) -> tuple[Dict | None, List[str]]:
        warnings: List[str] = []
        if not isinstance(changelog, dict):
            warnings.append("Changelog is not a JSON object.")
            return None, warnings

        required_keys = {"splits", "replacements", "added", "removed"}
        missing_keys = required_keys - set(changelog.keys())
        if missing_keys:
            warnings.append(
                f"Changelog missing required keys: {', '.join(sorted(missing_keys))}"
            )

        def map_id(value: str | None) -> str | None:
            if not value:
                return None
            return mapping.get(value, value)

        def parse_split_text(value: str) -> Dict | None:
            match = re.match(r"\s*(REQ-\d+)\s*->\s*(.+)", value)
            if not match:
                warnings.append(f"Unparseable split entry: {value}")
                return None
            from_id = match.group(1).strip()
            into_raw = match.group(2)
            into_ids = [item.strip() for item in into_raw.split(",") if item.strip()]
            if not into_ids:
                warnings.append(f"Split entry missing targets: {value}")
                return None
            return {"from": from_id, "into": into_ids}

        def normalize_split_entry(entry: object) -> Dict | None:
            if isinstance(entry, dict):
                from_id = entry.get("from")
                if not isinstance(from_id, str):
                    warnings.append(f"Split entry missing string from id: {entry}")
                    return None
                into_ids = entry.get("into", entry.get("to", []))
                if isinstance(into_ids, str):
                    into_ids = [into_ids]
                if not isinstance(into_ids, list):
                    warnings.append(f"Split entry has invalid targets: {entry}")
                    return None
                mapped_from = map_id(from_id)
                mapped_into = [
                    map_id(item) for item in into_ids if isinstance(item, str)
                ]
                mapped_into = [item for item in mapped_into if item]
                if not mapped_into:
                    warnings.append(f"Split entry missing targets: {entry}")
                    return None
                return {"from": mapped_from or from_id, "into": mapped_into}
            if isinstance(entry, str):
                parsed = parse_split_text(entry)
                if not parsed:
                    return None
                return {
                    "from": map_id(parsed["from"]) or parsed["from"],
                    "into": [map_id(item) or item for item in parsed["into"]],
                }
            warnings.append(f"Split entry has unsupported type: {entry}")
            return None

        raw_splits = changelog.get("splits", [])
        if isinstance(raw_splits, str):
            raw_splits = [raw_splits]
        if not isinstance(raw_splits, list):
            warnings.append("Splits entry is not a list or string.")
            raw_splits = []

        splits: List[Dict] = []
        for split in raw_splits:
            normalized = normalize_split_entry(split)
            if normalized:
                splits.append(normalized)

        def normalize_id_list(key: str) -> List[str]:
            values = changelog.get(key, [])
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list):
                warnings.append(f"{key} entry is not a list or string.")
                return []
            normalized: List[str] = []
            for item in values:
                if isinstance(item, str):
                    mapped = map_id(item)
                    if mapped:
                        normalized.append(mapped)
                elif isinstance(item, dict):
                    target = item.get("to") or item.get("id")
                    if isinstance(target, str):
                        mapped = map_id(target)
                        if mapped:
                            normalized.append(mapped)
                    else:
                        warnings.append(f"{key} entry has invalid object: {item}")
                else:
                    warnings.append(f"{key} entry has unsupported type: {item}")
            return normalized

        normalized = {
            "splits": splits,
            "replacements": normalize_id_list("replacements"),
            "added": normalize_id_list("added"),
            "removed": normalize_id_list("removed"),
        }
        if warnings:
            normalized["warnings"] = warnings
        return normalized, warnings

    def _merge_requirements(self, base: Dict, repair: Dict) -> Dict:
        base_items = list(base.get("requirements", []))
        seen = {(item.get("text"), item.get("priority")) for item in base_items}
        for item in repair.get("requirements", []):
            key = (item.get("text"), item.get("priority"))
            if key not in seen:
                seen.add(key)
                base_items.append(item)

        assumptions = list(base.get("assumptions", []))
        for item in repair.get("assumptions", []):
            if item not in assumptions:
                assumptions.append(item)

        constraints = list(base.get("constraints", []))
        for item in repair.get("constraints", []):
            if item not in constraints:
                constraints.append(item)

        return {
            "requirements": base_items,
            "assumptions": assumptions,
            "constraints": constraints,
        }

    def _merge_changelog(self, base: Dict, repair: Dict) -> Dict:
        def merge_list(key: str) -> List:
            merged = list(base.get(key, []))
            for item in repair.get(key, []):
                if item not in merged:
                    merged.append(item)
            return merged

        return {
            "splits": merge_list("splits"),
            "replacements": merge_list("replacements"),
            "added": merge_list("added"),
            "removed": merge_list("removed"),
            "warnings": merge_list("warnings"),
        }

    def _recompute_added(self, changelog: Dict, requirements: Dict, review: Dict) -> Dict:
        accepted_ids = set(review.get("accepted", []))
        final_ids = {item.get("id") for item in requirements.get("requirements", [])}
        added = sorted([item for item in final_ids if item not in accepted_ids])
        changelog["added"] = added
        return changelog

    def _record_extraction(self, note: str) -> None:
        self._extraction_traces.append(note)
        if self._env("ORCH_DEBUG_EXTRACT", "") == "1":
            print(f"[extract] {note}")

    def _validate_changelog(self, changelog: Dict) -> None:
        required_keys = {"splits", "replacements", "added", "removed"}
        if not isinstance(changelog, dict) or not required_keys.issubset(changelog.keys()):
            raise ValueError("Changelog JSON missing required keys.")

    def _limits_from_frontmatter(self, frontmatter: Dict) -> RequirementsLimits:
        req_target = frontmatter.get("requirements_target", {}) if isinstance(frontmatter, dict) else {}
        targets = frontmatter.get("targets", {}) if isinstance(frontmatter, dict) else {}
        if not isinstance(targets, dict):
            targets = {}
        req_min_value = targets.get(
            "target_min_items", frontmatter.get("target_min_reqs", req_target.get("min", 30))
        )
        req_max_value = targets.get(
            "target_max_items", frontmatter.get("target_max_reqs", req_target.get("max"))
        )
        try:
            req_min = int(req_min_value) if req_min_value is not None else 30
        except (TypeError, ValueError):
            req_min = 30
        req_max = req_max_value
        if req_max == "" or req_max is None:
            req_max_int = None
        else:
            try:
                req_max_int = int(req_max)
            except (TypeError, ValueError):
                req_max_int = None
        coverage_areas = frontmatter.get("coverage_areas", [])
        if isinstance(coverage_areas, str):
            coverage_areas = [coverage_areas]
        normalized_areas: List[str] = []
        if isinstance(coverage_areas, list):
            for entry in coverage_areas:
                if isinstance(entry, str):
                    normalized_areas.append(entry)
                elif isinstance(entry, dict):
                    name = entry.get("name")
                    if isinstance(name, str):
                        normalized_areas.append(name)
        coverage_areas = normalized_areas
        min_per_area = frontmatter.get("min_per_area")
        min_per_area_int = int(min_per_area) if min_per_area is not None else None
        seed_requirements = frontmatter.get("seed_requirements", [])
        if isinstance(seed_requirements, str):
            seed_requirements = [seed_requirements]
        normalized_seeds: List[str] = []
        for seed in seed_requirements if isinstance(seed_requirements, list) else []:
            if isinstance(seed, str):
                normalized_seeds.append(seed)
            elif isinstance(seed, dict):
                text = seed.get("text")
                seed_id = seed.get("id")
                if isinstance(seed_id, str) and isinstance(text, str):
                    normalized_seeds.append(f"{seed_id}: {text}")
                elif isinstance(text, str):
                    normalized_seeds.append(text)
        seed_requirements = normalized_seeds
        requested_artifacts = frontmatter.get("requested_artifacts", [])
        if isinstance(requested_artifacts, str):
            requested_artifacts = [requested_artifacts]
        if requested_artifacts:
            requested_list = [str(item).lower() for item in requested_artifacts]
        else:
            requested_list = [
                "requirements",
                "business_rules",
                "workflows",
                "domain_model",
                "mvp_scope",
                "acceptance_criteria",
            ]
        default_budgets = {
            "requirements": 2400,
            "business_rules": 1600,
            "workflows": 2000,
            "domain_model": 1600,
            "mvp_scope": 1200,
            "acceptance_criteria": 2000,
        }
        frontmatter_budgets = frontmatter.get("artifact_token_budgets", {})
        if not isinstance(frontmatter_budgets, dict):
            frontmatter_budgets = {}
        artifact_token_budgets: Dict[str, int] = {}
        for key, default_value in default_budgets.items():
            raw_value = frontmatter_budgets.get(key, frontmatter.get(f"{key}_max_output_tokens"))
            try:
                artifact_token_budgets[key] = int(raw_value) if raw_value is not None else default_value
            except (TypeError, ValueError):
                artifact_token_budgets[key] = default_value
        min_assumptions = targets.get("min_assumptions", frontmatter.get("assumptions_min", 3))
        min_constraints = targets.get("min_constraints", frontmatter.get("constraints_min", 3))
        try:
            assumptions_min = int(min_assumptions)
        except (TypeError, ValueError):
            assumptions_min = 3
        try:
            constraints_min = int(min_constraints)
        except (TypeError, ValueError):
            constraints_min = 3

        coverage_keywords: Dict[str, List[str]] = {}
        if isinstance(frontmatter.get("coverage_areas"), list):
            for entry in frontmatter["coverage_areas"]:
                if isinstance(entry, dict):
                    name = entry.get("name")
                    keywords = entry.get("keywords", [])
                    if isinstance(name, str):
                        if not isinstance(keywords, list):
                            keywords = []
                        coverage_keywords[name] = [str(item) for item in keywords if str(item)]

        return RequirementsLimits(
            req_min=req_min,
            req_max=req_max_int,
            assumptions_min=assumptions_min,
            constraints_min=constraints_min,
            roles_expected=list(frontmatter.get("roles_expected", [])),
            coverage_areas=coverage_areas,
            coverage_keywords=coverage_keywords,
            min_per_area=min_per_area_int,
            seed_requirements=seed_requirements,
            requested_artifacts=requested_list,
            artifact_token_budgets=artifact_token_budgets,
        )

    def _limits_payload(self, limits: RequirementsLimits) -> Dict:
        return {
            "requirements_min": limits.req_min,
            "requirements_max": limits.req_max,
            "assumptions_min": limits.assumptions_min,
            "constraints_min": limits.constraints_min,
            "roles_expected": limits.roles_expected,
            "coverage_areas": limits.coverage_areas,
            "min_per_area": limits.min_per_area,
            "seed_requirements": limits.seed_requirements,
        }

    def _metrics(self, requirements: Dict, limits: RequirementsLimits) -> Dict:
        return {
            "requirements_count": len(requirements.get("requirements", [])),
            "assumptions_count": len(requirements.get("assumptions", [])),
            "constraints_count": len(requirements.get("constraints", [])),
            "target_min": limits.req_min,
            "target_max": limits.req_max,
        }

    def _render_prompt(self, template: str, limits: RequirementsLimits) -> str:
        roles = ", ".join(limits.roles_expected) if limits.roles_expected else "none"
        coverage = ", ".join(limits.coverage_areas) if limits.coverage_areas else "none"
        seeds = "\n".join(f"- {seed}" for seed in limits.seed_requirements) or "none"
        min_per_area = (
            str(limits.min_per_area) if limits.min_per_area is not None else "none"
        )
        return (
            template.replace("{{REQ_MIN}}", str(limits.req_min))
            .replace("{{REQ_MAX}}", str(limits.req_max) if limits.req_max is not None else "none")
            .replace("{{ASSUMPTIONS_MIN}}", str(limits.assumptions_min))
            .replace("{{CONSTRAINTS_MIN}}", str(limits.constraints_min))
            .replace("{{ROLES_EXPECTED}}", roles)
            .replace("{{COVERAGE_AREAS}}", coverage)
            .replace("{{MIN_PER_AREA}}", min_per_area)
            .replace("{{SEED_REQUIREMENTS}}", seeds)
        )

    def _parse_frontmatter(self, content: str) -> Tuple[Dict, str]:
        if not content.startswith("---"):
            stripped = content.lstrip()
            if stripped.startswith("{"):
                try:
                    decoder = json.JSONDecoder()
                    parsed, end = decoder.raw_decode(stripped)
                except json.JSONDecodeError:
                    return {}, content
                if isinstance(parsed, dict):
                    body = stripped[end:].lstrip("\n")
                    return parsed, body
            return {}, content
        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}, content
        meta_raw = parts[1].strip()
        body = parts[2].lstrip("\n")
        try:
            meta = yaml.safe_load(meta_raw) or {}
        except yaml.YAMLError:
            meta = {}
        return meta, body

    def _gate_config(self) -> Dict:
        return {
            "min_count": int(self._env("ORCH_REQ_MIN_COUNT", "30")),
            "max_count": self._env("ORCH_REQ_MAX_COUNT", ""),
            "assumptions_min": int(self._env("ORCH_ASSUMPTIONS_MIN", "3")),
            "constraints_min": int(self._env("ORCH_CONSTRAINTS_MIN", "3")),
        }

    def _write_run_summary(
        self,
        artifacts_dir: Path,
        requirements: Dict,
        final_artifacts: Dict[str, Dict],
        acceptance_criteria: Dict | None,
        coverage: Dict,
        repairs_applied: bool,
        responses: List[LLMResponse],
    ) -> None:
        req_count = len(requirements.get("requirements", []))
        missing_areas = coverage.get("missing_areas", [])
        missing_seeds = coverage.get("missing_seeds", [])
        usage_totals = self._collect_usage_totals(artifacts_dir.parent / "raw", responses)
        lines = [
            "# Run Summary",
            "",
            f"- req_count: {req_count}",
            f"- missing_areas: {', '.join(missing_areas) if missing_areas else 'none'}",
            f"- missing_seeds: {', '.join(missing_seeds) if missing_seeds else 'none'}",
            f"- repairs_applied: {'yes' if repairs_applied else 'no'}",
        ]
        if final_artifacts.get("business_rules"):
            lines.append(f"- business_rules_count: {len(final_artifacts['business_rules'].get('rules', []))}")
        if final_artifacts.get("workflows"):
            lines.append(f"- workflows_count: {len(final_artifacts['workflows'].get('workflows', []))}")
        if final_artifacts.get("domain_model"):
            lines.append(f"- domain_entities_count: {len(final_artifacts['domain_model'].get('entities', []))}")
        if final_artifacts.get("mvp_scope"):
            lines.append(f"- mvp_in_scope_count: {len(final_artifacts['mvp_scope'].get('in_scope', []))}")
        stage_b_ok = not self._section_warnings.get("business_rules") and not self._section_warnings.get("workflows")
        stage_c_ok = not self._section_warnings.get("domain_model") and not self._section_warnings.get("mvp_scope")
        lines.append(f"- stage_b_success: {'yes' if stage_b_ok else 'no'}")
        lines.append(f"- stage_c_success: {'yes' if stage_c_ok else 'no'}")
        if acceptance_criteria:
            lines.append(f"- acceptance_criteria_count: {len(acceptance_criteria.get('criteria', []))}")
        if self._section_warnings:
            missing = ", ".join(
                key for key, items in self._section_warnings.items() if items
            )
            if missing:
                lines.append(f"- missing_sections: {missing}")
        if self._requirements_warnings:
            lines.append(
                f"- requirements_repairs: {len(self._requirements_warnings)} "
                "(see artifacts/warnings.json)"
            )
        if any(value > 0 for value in self._list_repair_counts.values()):
            lines.append(
                "- list_repairs: "
                + ", ".join(
                    f"{key}={value}" for key, value in self._list_repair_counts.items()
                )
            )
        if self._artifact_repair_counts:
            lines.append(
                "- artifact_repairs: "
                + ", ".join(
                    f"{key}={value}" for key, value in self._artifact_repair_counts.items()
                )
            )
        if self._artifact_validation:
            lines.append(
                "- artifact_validation: "
                + ", ".join(
                    f"{key}={value}" for key, value in self._artifact_validation.items()
                )
            )
        if self._delta_retry_counts:
            lines.append(
                "- delta_retries: "
                + ", ".join(
                    f"{key}={value}" for key, value in self._delta_retry_counts.items()
                )
            )
        if usage_totals:
            lines.append(
                "- token_usage: "
                + ", ".join(f"{key}={value}" for key, value in usage_totals.items())
            )
        if self._repair_warnings:
            lines.append(
                f"- repairs_warnings: {len(self._repair_warnings)} "
                "(see artifacts/repairs_warnings.json)"
            )
        if self._acceptance_warnings:
            lines.append(
                f"- acceptance_warnings: {len(self._acceptance_warnings)} "
                "(see artifacts/acceptance_warnings.json)"
            )
        write_text(artifacts_dir / "run_summary.md", "\n".join(lines) + "\n")

    def _collect_usage_totals(
        self, raw_dir: Path, responses: List[LLMResponse]
    ) -> Dict[str, int]:
        totals: Dict[str, int] = {}
        usage_files = list(raw_dir.glob("*_usage.json"))
        for path in usage_files:
            try:
                usage = json.loads(read_text(path))
            except Exception:
                continue
            if isinstance(usage, dict):
                for key, value in usage.items():
                    if isinstance(value, int):
                        totals[key] = totals.get(key, 0) + value
        if totals:
            return totals

        for response in responses:
            usage = getattr(response, "usage", None)
            if not usage:
                continue
            for key, value in usage.items():
                if isinstance(value, int):
                    totals[key] = totals.get(key, 0) + value
        return totals

    def _write_usage(self, path: Path, response: LLMResponse) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            write_json(path, usage)

    def _env(self, key: str, default: str) -> str:
        return os.getenv(key, default)

    def _load_schema(self, name: str) -> Dict:
        return json.loads(read_text(self.schemas_dir / name))
