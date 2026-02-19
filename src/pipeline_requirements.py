from __future__ import annotations

import json
import math
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from jsonschema import ValidationError, validate

from src.adapters.gemini_adapter import GeminiAdapter, GeminiUnavailableError
from src.adapters.llm_base import LLMAdapter, LLMResponse
from src.adapters.mock_adapter import MockAdapter
from src.adapters.openai_adapter import OpenAIAdapter
from src.artifacts.adr_writer import write_adr
from src.artifacts.writers import write_requirements
from src.gates.parsers import extract_json, extract_json_tolerant
from src.utils.io import read_text, write_json, write_text


class RequirementsFormatError(ValueError):
    pass


def parse_json_loose(text: str) -> Dict:
    parsed, repairs = _parse_json_loose_with_repairs(text)
    parse_json_loose.last_repairs = repairs
    return parsed


parse_json_loose.last_repairs = []


def _parse_json_loose_with_repairs(text: str) -> tuple[Dict, List[str]]:
    repairs: List[str] = []
    cleaned = text.strip()
    if cleaned != text:
        repairs.append("trimmed_whitespace")

    lines = cleaned.splitlines()
    stripped_lines = [line for line in lines if not line.lstrip().startswith("```")]
    if stripped_lines != lines:
        repairs.append("fences_removed")
        cleaned = "\n".join(stripped_lines)

    before_quotes = cleaned
    cleaned = cleaned.replace("\u201c", "\"").replace("\u201d", "\"").replace("\u2019", "'")
    if cleaned != before_quotes:
        repairs.append("smart_quotes_normalized")

    before_controls = cleaned
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", cleaned)
    if cleaned != before_controls:
        repairs.append("control_chars_removed")

    extracted = _extract_first_json_object_static(cleaned)
    if extracted and extracted != cleaned:
        repairs.append("balanced_json_extracted")
        cleaned = extracted

    cleaned_no_trailing = re.sub(r",\s*([}\]])", r"\1", cleaned)
    if cleaned_no_trailing != cleaned:
        repairs.append("trailing_commas_removed")
        cleaned = cleaned_no_trailing

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        snippet = cleaned.strip().replace("\n", " ")
        snippet = (snippet[:200] + "...") if len(snippet) > 200 else snippet
        raise ValueError(f"Loose JSON parse failed. Snippet: {snippet}. Error: {exc}") from exc
    if not isinstance(parsed, dict):
        snippet = cleaned.strip().replace("\n", " ")
        snippet = (snippet[:200] + "...") if len(snippet) > 200 else snippet
        raise ValueError(f"Loose JSON parse must return object. Snippet: {snippet}")
    return parsed, repairs


def _extract_first_json_object_static(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == "\"":
            in_string = not in_string
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


@dataclass
class RequirementsLimits:
    req_min: int
    req_max: int | None
    final_target_items: int | None
    add_only_batch_size: int
    add_only_max_rounds: int
    add_only_min_new_per_area: int | None
    assumptions_min: int
    constraints_min: int
    min_student_reqs: int
    min_coordinator_reqs: int
    min_admin_reqs: int
    min_domain_keyword_hits: int
    roles_expected: List[str]
    coverage_areas: List[str]
    coverage_keywords: Dict[str, List[str]]
    min_per_area: int | None
    coverage_prefix_mode: bool
    seed_requirements: List[str]
    requested_artifacts: List[str]
    artifact_token_budgets: Dict[str, int]
    lead_token_budgets: Dict[str, int]
    apply_token_budgets: Dict[str, int]


class RequirementsPipeline:
    _DOMAIN_KEYWORDS = [
        "mobility",
        "procedure",
        "deadline",
        "document",
        "exception",
        "approval",
        "rejection",
        "resubmit",
    ]
    _DEFAULT_COVERAGE_KEYWORDS: Dict[str, List[str]] = {
        "Identity & Authentication": ["identity", "authentication", "login", "logout", "mfa"],
        "RBAC & Permissions": ["rbac", "role", "permission", "access control", "authorization"],
        "Procedures & Publishing": ["procedure", "version", "publish", "draft", "template"],
        "Deadlines & Calendar": ["deadline", "calendar", "schedule", "due date", "timeline"],
        "Documents & Signatures": [
            "document",
            "upload",
            "validate",
            "signature",
            "sign",
            "attachment",
        ],
        "Status & Workflow": ["status", "workflow", "state", "stage", "lifecycle"],
        "Notifications": ["notification", "email", "sms", "push", "alert"],
        "Admin & Moderation": ["admin", "configuration", "moderation", "settings"],
        "Audit & Traceability": ["audit", "log", "trace", "history", "audit trail"],
        "Integrations": ["integration", "api", "sso", "webhook", "esign"],
        "Privacy & Retention": ["privacy", "gdpr", "retention", "consent", "data policy"],
        "Accessibility & I18n": ["accessibility", "wcag", "i18n", "localization", "a11y"],
        "Observability & SLAs": ["observability", "metrics", "alerts", "sla", "monitoring"],
    }

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
        self._requirements_shape_normalized = False
        self._requirements_filtered_out: List[Dict] = []
        self._requirements_quality_warnings: List[Dict] = []
        self._requirements_balance_results: Dict[str, object] = {}
        self._coverage_area_terms: List[str] = []
        self._out_of_scope_terms: List[str] = []
        self._requirements_filler_filtered: List[Dict] = []
        self._requirements_duplicates_debug: List[Dict] = []
        self._coverage_fix_used = False
        self._assumptions_added = 0
        self._constraints_added = 0
        self._add_only_parse_failures = 0
        self._apply_format_retry_used = False
        self._gemini_review_present = False
        self._gemini_review_used = False
        self._gemini_final_review_used = False
        self._post_review_add_only_used = False
        self._final_review_retry_used = False
        self._coverage_unmapped_count = 0
        self._add_only_completion_tokens: List[int] = []
        self._format_fix_completion_tokens: List[int] = []
        self._lead_completion_tokens: int | None = None
        self._apply_completion_tokens: int | None = None
        self._add_only_batch_size_used = 12
        self._single_requirement_fallback: Dict | None = None
        self._apply_action_retry_used = False
        self._extraction_debug: List[Dict[str, object]] = []
        self._json_parse_repairs: List[Dict[str, object]] = []
        self._draft_extracted_candidate: Dict | None = None
        self._draft_extracted_cleaned: Dict | None = None
        self._draft_candidate_before_repair_text: str | None = None
        self._draft_candidate_after_repair_text: str | None = None
        self._gemini_selected_model: str | None = None
        self._gemini_cross_review_skipped = False
        self._gemini_error_summary: str | None = None
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
        self._requirements_shape_normalized = False
        self._requirements_filtered_out = []
        self._requirements_quality_warnings = []
        self._requirements_balance_results = {}
        self._coverage_area_terms = []
        self._out_of_scope_terms = []
        self._requirements_filler_filtered = []
        self._requirements_duplicates_debug = []
        self._coverage_fix_used = False
        self._assumptions_added = 0
        self._constraints_added = 0
        self._add_only_parse_failures = 0
        self._apply_format_retry_used = False
        self._gemini_review_present = False
        self._gemini_review_used = False
        self._gemini_final_review_used = False
        self._post_review_add_only_used = False
        self._final_review_retry_used = False
        self._coverage_unmapped_count = 0
        self._single_requirement_fallback = None
        self._apply_action_retry_used = False
        self._extraction_debug = []
        self._json_parse_repairs = []
        self._draft_extracted_candidate = None
        self._draft_extracted_cleaned = None
        self._draft_candidate_before_repair_text = None
        self._draft_candidate_after_repair_text = None
        self._gemini_selected_model = None
        self._gemini_cross_review_skipped = False
        self._gemini_error_summary = None
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
        self._coverage_area_terms = list(limits.coverage_areas)
        self._out_of_scope_terms = self._out_of_scope_from_frontmatter(frontmatter)
        self._coverage_fix_used = self._coverage_defaults_used(frontmatter)

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

    def _stage_budget(
        self, limits: RequirementsLimits, artifact: str, stage: str, default_budget: int
    ) -> int:
        if stage == "lead":
            value = limits.lead_token_budgets.get(artifact, default_budget)
        elif stage == "apply":
            value = limits.apply_token_budgets.get(artifact, default_budget)
        else:
            value = default_budget
        try:
            return int(value)
        except (TypeError, ValueError):
            return default_budget

    def _stage_max_tokens(
        self, limits: RequirementsLimits, artifact: str, stage: str, default_budget: int
    ) -> int:
        max_value = self._stage_budget(limits, artifact, stage, default_budget)
        return self._apply_cli_cap(max_value)

    def _apply_cli_cap(self, max_value: int) -> int:
        cap_raw = self._env("ORCH_MAX_OUTPUT_TOKENS", "")
        cap_value: int | None = None
        if isinstance(cap_raw, str) and cap_raw.strip():
            try:
                cap_value = int(cap_raw)
            except (TypeError, ValueError):
                cap_value = None
        if cap_value and cap_value > 0:
            return min(max_value, cap_value)
        return max_value

    def _complete(
        self, adapter: LLMAdapter, prompt: str, max_tokens: int | None
    ) -> LLMResponse:
        return adapter.complete(prompt, max_tokens=max_tokens)

    def _completion_tokens(self, response: LLMResponse) -> int | None:
        usage = getattr(response, "usage", None)
        if isinstance(usage, dict):
            completion = usage.get("completion_tokens")
            if isinstance(completion, int):
                return completion
        return None

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
        base_budget = self._artifact_token_budget(artifact, limits)
        lead_budget = self._stage_budget(limits, artifact, "lead", base_budget)
        apply_budget = self._stage_budget(limits, artifact, "apply", base_budget)
        lead_tokens = self._stage_max_tokens(limits, artifact, "lead", base_budget)
        apply_tokens = self._stage_max_tokens(limits, artifact, "apply", base_budget)
        responses: List[LLMResponse] = []
        summary: Dict[str, object] = {}

        lead_template = read_text(self.prompts_dir / config["lead_prompt"])
        lead_prompt = self._render_prompt(lead_template, limits)
        lead_payload = {"brief": brief}
        lead_full_prompt = f"{lead_prompt}\n\nINPUT:\n{json.dumps(lead_payload)}\n"
        write_text(raw_dir / f"{artifact}_draft_prompt.txt", lead_full_prompt)
        lead_response = self._complete(chatgpt, lead_full_prompt, lead_tokens)
        self._lead_completion_tokens = self._completion_tokens(lead_response)
        responses.append(lead_response)
        write_text(raw_dir / f"{artifact}_draft_raw.txt", lead_response.raw_text)
        self._write_usage(raw_dir / f"{artifact}_draft_usage.json", lead_response)

        try:
            draft_payload = self._extract_wrapped_json(
                lead_response.raw_text,
                config["draft_label"],
                config["expected_keys"],
                context=f"{artifact}_draft",
            )
        except ValueError as exc:
            if artifact != "requirements":
                raise
            retry_prompt = read_text(self.prompts_dir / "requirements_lead_format_retry.md")
            retry_payload = {
                "parse_error": str(exc),
                "raw_output": lead_response.raw_text,
            }
            retry_full_prompt = f"{retry_prompt}\n\nINPUT:\n{json.dumps(retry_payload)}\n"
            write_text(
                raw_dir / "requirements_draft_format_retry_prompt.txt",
                retry_full_prompt,
            )
            retry_response = self._complete(chatgpt, retry_full_prompt, lead_tokens)
            responses.append(retry_response)
            write_text(
                raw_dir / "requirements_draft_format_retry_raw.txt",
                retry_response.raw_text,
            )
            self._write_usage(
                raw_dir / "requirements_draft_format_retry_usage.json", retry_response
            )
            try:
                draft_payload = self._extract_wrapped_json(
                    retry_response.raw_text,
                    config["draft_label"],
                    config["expected_keys"],
                    context=f"{artifact}_draft_retry",
                )
            except ValueError as retry_exc:
                raise RuntimeError(
                    "Requirements draft format retry failed to parse."
                ) from retry_exc
        if artifact == "requirements":
            write_json(artifacts_dir / "requirements_draft_extracted.json", draft_payload)
            if self._draft_extracted_candidate is not None:
                write_json(
                    artifacts_dir / "requirements_draft_extracted_candidate.json",
                    self._draft_extracted_candidate,
                )
            if self._draft_extracted_cleaned is not None:
                write_json(
                    artifacts_dir / "requirements_draft_extracted_cleaned.json",
                    self._draft_extracted_cleaned,
                )
            if self._draft_candidate_before_repair_text is not None:
                write_text(
                    artifacts_dir / "requirements_draft_candidate_before_repair.json",
                    self._draft_candidate_before_repair_text,
                )
            if self._draft_candidate_after_repair_text is not None:
                write_text(
                    artifacts_dir / "requirements_draft_candidate_after_repair.json",
                    self._draft_candidate_after_repair_text,
                )
        draft_payload, draft_warnings = self._repair_artifact_payload(
            artifact, draft_payload, stage="draft"
        )
        if artifact == "requirements":
            write_json(artifacts_dir / "requirements_draft_normalized.json", draft_payload)

        if artifact == "requirements":
            cross_template = read_text(self.prompts_dir / "requirements_cross_review.md")
            cross_review_prompt = self._render_prompt(cross_template, limits)
            cross_payload = {
                "brief": brief,
                "requirements": draft_payload,
                "targets": self._requirements_targets_payload(limits),
            }
        else:
            cross_review_prompt = self._artifact_cross_review_prompt(artifact)
            cross_payload = {"brief": brief, "artifact": draft_payload}
        cross_full_prompt = f"{cross_review_prompt}\n\nINPUT:\n{json.dumps(cross_payload)}\n"
        write_text(raw_dir / f"{artifact}_cross_review_prompt.txt", cross_full_prompt)
        cross_review_error: str | None = None
        cross_review_skipped = False
        cross_response: LLMResponse | None = None
        try:
            cross_response = self._complete(gemini, cross_full_prompt, apply_tokens)
        except (GeminiUnavailableError, RuntimeError) as exc:
            cross_review_error = str(exc)
            cross_review_skipped = True
            self._gemini_cross_review_skipped = True
            self._gemini_error_summary = cross_review_error
            diagnostics = {}
            if hasattr(gemini, "get_diagnostics"):
                try:
                    diagnostics = gemini.get_diagnostics()  # type: ignore[attr-defined]
                except Exception:
                    diagnostics = {}
            if artifact == "requirements":
                write_json(
                    artifacts_dir / "requirements_cross_review_failed.json",
                    {
                        "error": cross_review_error,
                        "cross_review_skipped": True,
                        "diagnostics": diagnostics,
                    },
                )
        if hasattr(gemini, "get_diagnostics"):
            try:
                diagnostics = gemini.get_diagnostics()  # type: ignore[attr-defined]
                selected = diagnostics.get("selected_model")
                if isinstance(selected, str) and selected:
                    self._gemini_selected_model = selected
            except Exception:
                pass
        cross_review: Dict = {}
        cross_review_parse_error: str | None = None
        if cross_response is not None:
            responses.append(cross_response)
            write_text(raw_dir / f"{artifact}_cross_review_raw.txt", cross_response.raw_text)
            self._write_usage(raw_dir / f"{artifact}_cross_review_usage.json", cross_response)
            cross_review = self._safe_extract_json(cross_response.raw_text)
            if artifact == "requirements":
                try:
                    cross_review = extract_json_tolerant(cross_response.raw_text)
                except ValueError as exc:
                    cross_review_parse_error = str(exc)
                    cross_review = {
                        "blocking_issues": [],
                        "required_actions": [],
                        "weak_requirements": [],
                        "missing_areas": [],
                    }
                    write_json(
                        artifacts_dir / "requirements_cross_review_failed.json",
                        {
                            "error": cross_review_parse_error,
                            "note": "Continuing without cross-review enforcement.",
                            "raw_snippet": cross_response.raw_text[:500],
                        },
                    )
                else:
                    write_json(
                        artifacts_dir / "requirements_cross_review_extracted.json", cross_review
                    )
                    cross_review = self._validate_requirements_review(
                        cross_review, draft_payload, limits
                    )
                    write_json(artifacts_dir / "requirements_gemini_review.json", cross_review)
                    write_json(
                        artifacts_dir / "requirements_cross_review_normalized.json", cross_review
                    )
                    self._gemini_review_present = True
        if artifact == "requirements" and cross_review_error:
            self._requirements_warnings.append(
                {
                    "stage": "cross_review",
                    "note": "Gemini cross-review failed; continuing without enforcement.",
                    "error": cross_review_error,
                }
            )
        if artifact == "requirements" and cross_review_parse_error:
            self._requirements_warnings.append(
                {
                    "stage": "cross_review",
                    "note": "Cross-review JSON parse failed; continuing without enforcement.",
                    "error": cross_review_parse_error,
                }
            )

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
            if cross_response is not None:
                apply_payload["gemini_review_text"] = cross_response.raw_text
        apply_instruction = ""
        if artifact == "requirements" and cross_review_error is None and cross_review_parse_error is None:
            missing_points = self._gemini_missing_points(cross_review)
            missing_points_list = (
                "\n".join(f"- {point}" for point in missing_points) if missing_points else "- none"
            )
            apply_instruction = (
                "\n\nYou may add NEW requirements to meet targets and missing coverage."
                "\n\nGemini missing points:\n"
                f"{missing_points_list}\n\nFor each missing point, add at least 2 new functional "
                "requirements (or modify existing ones) that directly address it."
                "\nMove NFR-like statements to constraints instead of requirements."
                "\nYou MUST satisfy each required_actions entry from the Gemini review JSON."
                "\nReport addressed_actions as an array of strings in ADDRESSED_ACTIONS_JSON."
            )
            if limits.coverage_prefix_mode:
                areas = ", ".join(limits.coverage_areas)
                apply_instruction += (
                    "\nPrefix each requirement with [<Coverage Area>] using one of: "
                    f"{areas}."
                )
            self._gemini_review_used = True
        apply_full_prompt = f"{apply_prompt}{apply_instruction}\n\nINPUT:\n{json.dumps(apply_payload)}\n"
        write_text(raw_dir / f"{artifact}_apply_prompt.txt", apply_full_prompt)
        apply_response = self._complete(chatgpt, apply_full_prompt, apply_tokens)
        self._apply_completion_tokens = self._completion_tokens(apply_response)
        responses.append(apply_response)
        write_text(raw_dir / f"{artifact}_apply_raw.txt", apply_response.raw_text)
        self._write_usage(raw_dir / f"{artifact}_apply_usage.json", apply_response)

        if artifact == "requirements":
            try:
                apply_extracted_raw = self._extract_wrapped_json(
                    apply_response.raw_text,
                    config["final_label"],
                    config["expected_keys"],
                    context="requirements_apply",
                )
                write_json(
                    artifacts_dir / "requirements_apply_extracted_raw.json",
                    apply_extracted_raw,
                )
                final_payload = apply_extracted_raw
            except RequirementsFormatError:
                final_payload = self._format_retry_requirements(
                    brief=brief,
                    apply_raw=apply_response.raw_text,
                    adapter=chatgpt,
                    raw_dir=raw_dir,
                    artifacts_dir=artifacts_dir,
                    max_tokens=apply_tokens,
                    expected_keys=config["expected_keys"],
                )
                write_json(
                    artifacts_dir / "requirements_apply_extracted_raw.json",
                    final_payload,
                )
                self._apply_format_retry_used = True
            except ValueError:
                truncation_detected = self._detect_truncation(apply_response.raw_text)
                if truncation_detected:
                    final_payload = self._format_fix_requirements(
                        brief=brief,
                        apply_raw=apply_response.raw_text,
                        adapter=chatgpt,
                        raw_dir=raw_dir,
                        artifacts_dir=artifacts_dir,
                        max_tokens=apply_tokens,
                        expected_keys=config["expected_keys"],
                    )
                    write_json(
                        artifacts_dir / "requirements_apply_extracted_raw.json",
                        final_payload,
                    )
                else:
                    raise
            try:
                addressed_actions = self._extract_wrapped_json_any(
                    apply_response.raw_text,
                    ["ADDRESSED_ACTIONS_JSON"],
                    {"addressed_actions"},
                )
                write_json(
                    artifacts_dir / "requirements_addressed_actions.json",
                    addressed_actions,
                )
                write_json(
                    artifacts_dir / "requirements_apply_log.json",
                    {
                        "required_actions": cross_review.get("required_actions", []),
                        "addressed_actions": addressed_actions.get("addressed_actions", []),
                    },
                )
            except ValueError:
                self._requirements_warnings.append(
                    {"stage": "apply", "note": "Missing addressed_actions JSON."}
                )
                write_json(
                    artifacts_dir / "requirements_apply_log.json",
                    {
                        "required_actions": cross_review.get("required_actions", []),
                        "addressed_actions": [],
                    },
                )
        else:
            final_payload = self._extract_wrapped_json(
                apply_response.raw_text,
                config["final_label"],
                config["expected_keys"],
            )
        changelog = None
        apply_report = None
        if artifact == "requirements":
            apply_report = self._extract_apply_report(
                apply_response.raw_text,
                artifacts_dir,
                stage="apply",
            )
            try:
                changelog = self._extract_wrapped_json(
                    apply_response.raw_text,
                    "CHANGELOG_JSON",
                    {"splits", "replacements", "added", "removed"},
                )
                write_json(raw_dir / "requirements_apply_changelog.json", changelog)
            except ValueError as exc:
                self._requirements_warnings.append(
                    {"stage": "apply", "note": "Missing changelog JSON.", "error": str(exc)}
                )
        final_payload, final_warnings = self._repair_artifact_payload(
            artifact, final_payload, stage="apply"
        )
        if artifact == "requirements":
            write_json(
                artifacts_dir / "requirements_apply_extracted_normalized.json",
                final_payload,
            )
            required_actions = cross_review.get("required_actions", [])
            blocking_issues = cross_review.get("blocking_issues", [])
            apply_report_errors: List[str] = []
            missing_actions: List[str] = []
            evidence_issues: List[Dict[str, str]] = []
            if apply_report is None and required_actions:
                apply_report_errors = [
                    "Missing APPLY_REPORT_JSON or ADDRESSED_ACTIONS_JSON."
                ]
                missing_actions = list(required_actions)
            elif apply_report is not None:
                (
                    apply_report_errors,
                    missing_actions,
                    evidence_issues,
                ) = self._validate_apply_report(
                    apply_report,
                    final_payload,
                    required_actions,
                )
            if blocking_issues and (missing_actions or evidence_issues):
                final_payload, apply_report = self._retry_apply_for_actions(
                    brief=brief,
                    draft=draft_payload,
                    cross_review=cross_review,
                    final_payload=final_payload,
                    apply_report=apply_report or {},
                    errors=apply_report_errors,
                    missing_actions=missing_actions,
                    evidence_issues=evidence_issues,
                    adapter=chatgpt,
                    raw_dir=raw_dir,
                    artifacts_dir=artifacts_dir,
                    max_tokens=apply_tokens,
                    expected_keys=config["expected_keys"],
                )
            elif apply_report_errors:
                self._requirements_warnings.append(
                    {
                        "stage": "apply",
                        "note": "Apply report missing or incomplete; enforcement skipped.",
                        "errors": apply_report_errors,
                    }
                )

        warnings = draft_warnings + final_warnings
        if warnings:
            self._artifact_repair_counts[artifact] = len(warnings)
            write_json(
                artifacts_dir / f"{artifact}_warnings.json",
                {"warnings": warnings},
            )

        retry_count = 0
        add_only_attempts = 0
        expand_generic_attempts = 0
        id_normalized = False
        missing_coverage_areas: List[str] = []
        missing_fields: List[str] = []
        schema = self._load_schema(config["schema"])
        if artifact == "requirements":
            final_payload, filtered_out, quality_warnings = self._apply_quality_gate(
                final_payload, limits
            )
            if filtered_out:
                self._requirements_filtered_out.extend(filtered_out)
            if quality_warnings:
                self._requirements_quality_warnings.extend(quality_warnings)
            count_before_add_only = len(final_payload.get("requirements", []))
            try:
                validate(instance=final_payload, schema=schema)
            except ValidationError as exc:
                self._requirements_warnings.append(
                    {
                        "stage": "apply",
                        "note": "Validation failed before add-only.",
                        "error": str(exc),
                    }
                )
            (
                final_payload,
                missing_coverage_areas,
                add_only_attempts,
                balance_results,
                add_only_requested,
                add_only_round_counts,
            ) = self._add_only_requirements_loop(
                brief=brief,
                limits=limits,
                payload=final_payload,
                adapter=chatgpt,
                gemini_review=cross_review,
                raw_dir=raw_dir,
                artifacts_dir=artifacts_dir,
                max_tokens=apply_tokens,
            )
            count_after_add_only = len(final_payload.get("requirements", []))
            missing_before_add_only = max(limits.req_min - count_before_add_only, 0)
            missing_after_add_only = max(limits.req_min - count_after_add_only, 0)
            final_payload, expand_generic_attempts = self._expand_generic_requirements(
                brief=brief,
                limits=limits,
                payload=final_payload,
                adapter=chatgpt,
                raw_dir=raw_dir,
                artifacts_dir=artifacts_dir,
                max_tokens=apply_tokens,
            )
            final_payload, assumptions_added, constraints_added = self._add_assumptions_constraints(
                brief=brief,
                limits=limits,
                payload=final_payload,
                adapter=chatgpt,
                raw_dir=raw_dir,
                artifacts_dir=artifacts_dir,
                max_tokens=apply_tokens,
            )
            assumptions_fixed = assumptions_added > 0
            constraints_fixed = constraints_added > 0
            self._assumptions_added += assumptions_added
            self._constraints_added += constraints_added
            final_payload, filtered_out, quality_warnings = self._apply_quality_gate(
                final_payload, limits, remove_items=False
            )
            if filtered_out:
                self._requirements_filtered_out.extend(filtered_out)
            if quality_warnings:
                self._requirements_quality_warnings.extend(quality_warnings)
            final_payload, self._post_review_add_only_used = self._run_final_review_add_only(
                brief=brief,
                limits=limits,
                payload=final_payload,
                adapter=chatgpt,
                gemini_adapter=gemini,
                cross_review=cross_review,
                raw_dir=raw_dir,
                artifacts_dir=artifacts_dir,
                max_tokens=apply_tokens,
                attempt_offset=add_only_attempts,
            )
            total_add_only_attempts = add_only_attempts + (
                1 if self._post_review_add_only_used else 0
            )
            balance_results = self._balance_check(final_payload, limits)
            self._requirements_balance_results = balance_results
            final_payload, id_normalized, id_map, changelog = self._normalize_requirement_ids(
                final_payload, changelog
            )
            if id_normalized:
                write_json(artifacts_dir / "requirements_id_map.json", {"id_map": id_map})
            if changelog is not None:
                write_json(artifacts_dir / "requirements_changelog.json", changelog)
            if apply_report is not None:
                if id_normalized:
                    apply_report = self._remap_apply_report_ids(apply_report, id_map)
                write_json(artifacts_dir / "requirements_apply_report.json", apply_report)
            coverage_counts = self._coverage_counts(final_payload, limits)
            cli_cap_raw = self._env("ORCH_MAX_OUTPUT_TOKENS", "")
            cli_cap = None
            if isinstance(cli_cap_raw, str) and cli_cap_raw.strip():
                try:
                    cli_cap = int(cli_cap_raw)
                except (TypeError, ValueError):
                    cli_cap = None
            summary.update(
                {
                    "gemini_cross_review_error": cross_review_error,
                    "gemini_cross_review_skipped": cross_review_skipped,
                    "gemini_selected_model": self._gemini_selected_model,
                    "gemini_error_summary": self._gemini_error_summary,
                    "cross_review_parse_error": cross_review_parse_error,
                    "lead_budget_max_output_tokens": lead_budget,
                    "apply_budget_max_output_tokens": apply_budget,
                    "lead_effective_max_output_tokens": lead_tokens,
                    "apply_effective_max_output_tokens": apply_tokens,
                    "lead_max_output_tokens": lead_tokens,
                    "apply_max_output_tokens": apply_tokens,
                    "initial_count": count_before_add_only,
                    "target_min_items": limits.req_min,
                    "final_target_items": limits.final_target_items,
                    "actual_count": len(final_payload.get("requirements", [])),
                    "assumptions_count": len(final_payload.get("assumptions", [])),
                    "constraints_count": len(final_payload.get("constraints", [])),
                    "missing_coverage_areas": missing_coverage_areas,
                    "add_only_attempts": add_only_attempts,
                    "total_add_only_attempts": total_add_only_attempts,
                    "missing_before_add_only": missing_before_add_only,
                    "missing_after_add_only": missing_after_add_only,
                    "count_before_add_only": count_before_add_only,
                    "count_after_add_only": count_after_add_only,
                    "add_only_chunk_size": self._add_only_batch_size_used,
                    "add_only_requested": add_only_requested,
                    "add_only_round_counts": add_only_round_counts,
                    "add_only_parse_failures": self._add_only_parse_failures,
                    "expand_generic_attempts": expand_generic_attempts,
                    "id_normalized": id_normalized,
                    "review_actions_applied": bool(cross_review.get("required_actions")),
                    "requirements_shape_normalized": self._requirements_shape_normalized,
                    "filtered_out_count": len(self._requirements_filtered_out),
                    "filler_filtered_count": len(self._requirements_filler_filtered),
                    "dedupe_count": len(self._requirements_duplicates_debug),
                    "balance_check_results": balance_results,
                    "coverage_counts": coverage_counts,
                    "coverage_fix_used": self._coverage_fix_used,
                    "assumptions_added": self._assumptions_added,
                    "constraints_added": self._constraints_added,
                    "assumptions_fixed": assumptions_fixed,
                    "constraints_fixed": constraints_fixed,
                    "apply_format_retry_used": self._apply_format_retry_used,
                    "gemini_review_present": self._gemini_review_present,
                    "gemini_review_used": self._gemini_review_used,
                    "gemini_final_review_used": self._gemini_final_review_used,
                    "post_review_add_only_used": self._post_review_add_only_used,
                    "final_review_retry_used": self._final_review_retry_used,
                    "apply_action_retry_used": self._apply_action_retry_used,
                    "min_enforcement_unmet": len(final_payload.get("requirements", []))
                    < limits.req_min,
                    "coverage_unmapped_count": self._coverage_unmapped_count,
                    "wrapper_repairs_applied": bool(
                        self._apply_format_retry_used or self._json_parse_repairs
                    ),
                    "cli_max_output_tokens": cli_cap,
                    "add_only_max_output_tokens": apply_tokens,
                    "format_retry_max_output_tokens": apply_tokens,
                    "assumptions_constraints_max_output_tokens": apply_tokens,
                    "effective_add_only_max_output_tokens": apply_tokens,
                    "effective_format_fix_max_output_tokens": apply_tokens,
                    "lead_completion_tokens": self._lead_completion_tokens,
                    "apply_completion_tokens": self._apply_completion_tokens,
                    "add_only_completion_tokens": list(self._add_only_completion_tokens),
                    "format_fix_completion_tokens": list(self._format_fix_completion_tokens),
                }
            )
            if self._requirements_filtered_out:
                write_json(
                    artifacts_dir / "requirements_filtered_out.json",
                    {"filtered_out": self._requirements_filtered_out},
                )
            if self._requirements_filler_filtered:
                write_json(
                    artifacts_dir / "filtered_out.json",
                    {"filtered_out": self._requirements_filler_filtered},
                )
            if self._requirements_quality_warnings:
                write_json(
                    artifacts_dir / "requirements_quality_warnings.json",
                    {"warnings": self._requirements_quality_warnings},
                )
            if self._requirements_duplicates_debug:
                write_json(
                    artifacts_dir / "requirements_duplicates_debug.json",
                    {"duplicates": self._requirements_duplicates_debug},
                )
            if self._json_parse_repairs:
                write_json(
                    artifacts_dir / "requirements_json_parse_repairs.json",
                    {"repairs": self._json_parse_repairs},
                )
            if self._extraction_debug:
                write_json(
                    artifacts_dir / "extraction_debug.json",
                    {"events": self._extraction_debug},
                )
            write_json(artifacts_dir / "coverage_counts.json", coverage_counts)
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
            retry_response = self._complete(chatgpt, retry_full_prompt, apply_tokens)
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
        self, raw_text: str, label: str, expected_keys: set[str], context: str | None = None
    ) -> Dict:
        attempts: List[str] = []
        last_snippet = raw_text.strip().replace("\n", " ")
        truncation_detected = self._detect_truncation(raw_text)
        debug_enabled = self._env("ORCH_DEBUG_EXTRACT", "0") == "1"

        def snippet_for(text: str) -> str:
            cleaned = text.strip().replace("\n", " ")
            return (cleaned[:200] + "...") if len(cleaned) > 200 else cleaned

        def unwrap_wrapper(candidate: Dict) -> Dict | None:
            if label in candidate and isinstance(candidate[label], dict):
                return candidate[label]
            lower_label = label.lower()
            for key, value in candidate.items():
                if isinstance(key, str) and key.lower() == lower_label and isinstance(value, dict):
                    return value
            if expected_keys.issubset(candidate.keys()):
                return candidate
            return None

        def handle_candidate(candidate: object, path: str, snippet: str) -> Dict | None:
            attempts.append(path)
            nonlocal last_snippet
            last_snippet = snippet
            if not isinstance(candidate, dict):
                if isinstance(candidate, list) and expected_keys == {"requirements", "assumptions", "constraints"}:
                    if all(
                        isinstance(item, dict) and self._is_requirement_object(item)
                        for item in candidate
                    ):
                        self._requirements_warnings.append(
                            {
                                "stage": "extract",
                                "note": "Requirements list returned without wrapper.",
                                "path": path,
                            }
                        )
                        return {
                            "requirements": candidate,
                            "assumptions": [],
                            "constraints": [],
                        }
                return None
            if self._is_requirement_object(candidate) and label not in candidate:
                if expected_keys == {"requirements", "assumptions", "constraints"}:
                    if context == "requirements_apply":
                        self._requirements_warnings.append(
                            {
                                "stage": "extract",
                                "note": "Single requirement object returned; triggering format retry.",
                                "path": path,
                            }
                        )
                        self._single_requirement_fallback = {
                            "requirements": [candidate],
                            "assumptions": [],
                            "constraints": [],
                        }
                        raise RequirementsFormatError(
                            f"{label} missing wrapper; found requirement object. Path: {path}. "
                            f"Snippet: {snippet}"
                        )
                    return {
                        "requirements": [candidate],
                        "assumptions": [],
                        "constraints": [],
                    }
                raise RequirementsFormatError(
                    f"{label} missing wrapper; found requirement object. Path: {path}. "
                    f"Snippet: {snippet}"
                )
            wrapper_value = unwrap_wrapper(candidate)
            if wrapper_value is not None:
                missing_keys = expected_keys.difference(wrapper_value.keys())
                if missing_keys:
                    keys = ", ".join(sorted(wrapper_value.keys()))
                    raise ValueError(
                        f"{label} wrapper missing keys: {', '.join(sorted(missing_keys))}. "
                        f"Found keys: {keys}. Path: {path}. Snippet: {snippet}"
                    )
                if (
                    expected_keys == {"requirements", "assumptions", "constraints"}
                    and isinstance(wrapper_value, dict)
                ):
                    self._normalize_requirement_ids(wrapper_value)
                return wrapper_value
            return None

        parse_error: str | None = None
        parsed = None

        decoded = self._raw_decode_json_object(raw_text)
        if decoded is not None:
            candidate = handle_candidate(decoded, "raw-decode", snippet_for(raw_text))
            if candidate is not None:
                self._record_extraction_debug(
                    context,
                    "raw-decode",
                    truncation_detected,
                    None,
                    debug_enabled,
                )
                return candidate
        else:
            attempts.append("raw-decode (parse-failed)")

        fenced = self._extract_fenced_json(raw_text)
        if fenced is not None:
            try:
                parsed = parse_json_loose(fenced)
                candidate = handle_candidate(parsed, "fenced-json", snippet_for(fenced))
                if candidate is not None:
                    self._record_parse_repairs(context, parse_json_loose.last_repairs)
                    self._record_extraction_debug(
                        context,
                        "fenced-json",
                        truncation_detected,
                        None,
                        debug_enabled,
                    )
                    return candidate
                self._record_parse_repairs(context, parse_json_loose.last_repairs)
            except ValueError as exc:
                parse_error = str(exc)
                attempts.append("fenced-json (parse-failed)")

        label_extracted = self._extract_json_after_label(raw_text, label)
        if label_extracted is not None:
            try:
                parsed = parse_json_loose(label_extracted)
                candidate = handle_candidate(parsed, "label-balanced", snippet_for(label_extracted))
                if candidate is not None:
                    self._record_parse_repairs(context, parse_json_loose.last_repairs)
                    self._record_extraction_debug(
                        context,
                        "label-balanced",
                        truncation_detected,
                        None,
                        debug_enabled,
                    )
                    return candidate
                self._record_parse_repairs(context, parse_json_loose.last_repairs)
            except ValueError as exc:
                parse_error = str(exc)
                attempts.append("label-balanced (parse-failed)")

        marker_extracted = self._extract_json_after_marker(raw_text, label)
        if marker_extracted is not None:
            try:
                parsed = parse_json_loose(marker_extracted)
                candidate = handle_candidate(parsed, "label-marker", snippet_for(marker_extracted))
                if candidate is not None:
                    self._record_parse_repairs(context, parse_json_loose.last_repairs)
                    self._record_extraction_debug(
                        context,
                        "label-marker",
                        truncation_detected,
                        None,
                        debug_enabled,
                    )
                    return candidate
                self._record_parse_repairs(context, parse_json_loose.last_repairs)
            except ValueError as exc:
                parse_error = str(exc)
                attempts.append("label-marker (parse-failed)")

        between_braces = self._extract_between_braces(raw_text)
        if between_braces is not None:
            try:
                parsed = parse_json_loose(between_braces)
                candidate = handle_candidate(parsed, "between-braces", snippet_for(between_braces))
                if candidate is not None:
                    self._record_parse_repairs(context, parse_json_loose.last_repairs)
                    self._record_extraction_debug(
                        context,
                        "between-braces",
                        truncation_detected,
                        None,
                        debug_enabled,
                    )
                    return candidate
                self._record_parse_repairs(context, parse_json_loose.last_repairs)
            except ValueError as exc:
                parse_error = str(exc)
                attempts.append("between-braces (parse-failed)")

        largest = self._largest_balanced_json(raw_text)
        if largest is not None:
            try:
                parsed = parse_json_loose(largest)
                candidate = handle_candidate(parsed, "largest-balanced", snippet_for(largest))
                if candidate is not None:
                    self._record_parse_repairs(context, parse_json_loose.last_repairs)
                    self._record_extraction_debug(
                        context,
                        "largest-balanced",
                        truncation_detected,
                        None,
                        debug_enabled,
                    )
                    return candidate
                self._record_parse_repairs(context, parse_json_loose.last_repairs)
            except ValueError as exc:
                parse_error = str(exc)
                attempts.append("largest-balanced (parse-failed)")
        else:
            attempts.append("largest-balanced (none)")

        if isinstance(parsed, dict) and "requirements" not in parsed:
            if self._is_requirement_object(parsed):
                if expected_keys == {"requirements", "assumptions", "constraints"}:
                    if context == "requirements_apply":
                        self._single_requirement_fallback = {
                            "requirements": [parsed],
                            "assumptions": [],
                            "constraints": [],
                        }
                        raise RequirementsFormatError(
                            f"{label} missing wrapper; found requirement object. "
                            f"Snippet: {snippet_for(last_snippet)}"
                        )
                    return {
                        "requirements": [parsed],
                        "assumptions": [],
                        "constraints": [],
                    }
        if isinstance(parsed, dict):
            missing = expected_keys.difference(parsed.keys())
            if not missing:
                self._record_extraction_debug(
                    context,
                    "payload-direct",
                    truncation_detected,
                    None,
                    debug_enabled,
                )
                return parsed

        if expected_keys == {"requirements", "assumptions", "constraints"}:
            tolerant_candidate = self._tolerant_json_extract(raw_text, attempts)
            if tolerant_candidate is not None:
                candidate = handle_candidate(
                    tolerant_candidate,
                    "tolerant-extract",
                    snippet_for(raw_text),
                )
                if candidate is not None:
                    self._record_extraction_debug(
                        context,
                        "tolerant-extract",
                        truncation_detected,
                        None,
                        debug_enabled,
                    )
                    return candidate

        self._record_extraction_debug(
            context,
            "failed",
            truncation_detected,
            parse_error,
            debug_enabled,
        )
        context_label = context or "unknown"
        raise ValueError(
            f"{label} extraction failed for {context_label}. "
            f"Attempts: {', '.join(attempts) or 'none'}. "
            f"Snippet: {snippet_for(last_snippet)}"
        )

    def _tolerant_json_extract(self, raw_text: str, attempts: List[str]) -> Dict | None:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            attempts.append("tolerant-extract (no-braces)")
            return None
        candidate_text = raw_text[start : end + 1]
        cleaned = (
            candidate_text.replace("\u201c", "\"")
            .replace("\u201d", "\"")
            .replace("\u2019", "'")
        )
        attempts.append("tolerant-extract (raw)")
        try:
            parsed = json.loads(cleaned)
            if self._draft_extracted_candidate is None:
                self._draft_extracted_candidate = parsed if isinstance(parsed, dict) else None
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError as exc:
            attempts.append(f"tolerant-extract (parse-failed: {exc})")
            if "Expecting ',' delimiter" in str(exc):
                repaired_text, repaired = self._insert_missing_commas_in_arrays(cleaned)
                if repaired:
                    self._draft_candidate_before_repair_text = cleaned
                    self._draft_candidate_after_repair_text = repaired_text
                    self._requirements_warnings.append(
                        {
                            "repair": "insert_missing_commas",
                            "path": "tolerant-extract",
                            "note": "Inserted commas between adjacent objects inside arrays.",
                        }
                    )
                    attempts.append("tolerant-extract (inserted-missing-commas)")
                    try:
                        parsed = json.loads(repaired_text)
                        if self._draft_extracted_candidate is None:
                            self._draft_extracted_candidate = (
                                parsed if isinstance(parsed, dict) else None
                            )
                        return parsed if isinstance(parsed, dict) else None
                    except json.JSONDecodeError as repair_exc:
                        attempts.append(
                            f"tolerant-extract (inserted-missing-commas-parse-failed: {repair_exc})"
                        )
        cleaned_commas = re.sub(r",\s*([}\]])", r"\1", cleaned)
        try:
            parsed = json.loads(cleaned_commas)
            if self._draft_extracted_candidate is None:
                self._draft_extracted_candidate = parsed if isinstance(parsed, dict) else None
            self._draft_extracted_cleaned = parsed if isinstance(parsed, dict) else None
            attempts.append("tolerant-extract (cleaned)")
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError as exc:
            attempts.append(f"tolerant-extract (cleaned-parse-failed: {exc})")
            return None

    def _insert_missing_commas_in_arrays(self, text: str) -> tuple[str, bool]:
        output: List[str] = []
        in_string = False
        escape = False
        array_depth = 0
        changed = False
        length = len(text)
        idx = 0
        while idx < length:
            ch = text[idx]
            if escape:
                output.append(ch)
                escape = False
                idx += 1
                continue
            if ch == "\\":
                output.append(ch)
                escape = True
                idx += 1
                continue
            if ch == "\"":
                in_string = not in_string
                output.append(ch)
                idx += 1
                continue
            if not in_string:
                if ch == "[":
                    array_depth += 1
                elif ch == "]":
                    array_depth = max(0, array_depth - 1)
            if not in_string and ch == "}" and array_depth > 0:
                output.append(ch)
                lookahead = idx + 1
                while lookahead < length and text[lookahead].isspace():
                    lookahead += 1
                if lookahead < length and text[lookahead] == "{":
                    output.append(",")
                    changed = True
                idx += 1
                continue
            output.append(ch)
            idx += 1
        return "".join(output), changed

    def _largest_json_object(self, raw_text: str) -> Dict[str, object] | None:
        decoder = json.JSONDecoder()
        best: Dict[str, object] | None = None
        idx = 0
        while idx < len(raw_text):
            start = self._find_next_json_start(raw_text, idx)
            if start == -1:
                break
            try:
                parsed, end = decoder.raw_decode(raw_text[start:])
            except json.JSONDecodeError:
                idx = start + 1
                continue
            snippet = raw_text[start : start + end]
            if isinstance(parsed, dict):
                if best is None or len(snippet) > len(str(best["snippet"])):
                    best = {"parsed": parsed, "snippet": snippet}
            idx = start + end
        return best

    def _extract_wrapped_json_any(
        self, raw_text: str, labels: List[str], expected_keys: set[str]
    ) -> Dict:
        payload = self._safe_extract_json(raw_text)
        if isinstance(payload, dict):
            for label in labels:
                if label in payload and isinstance(payload[label], dict):
                    payload = payload[label]
                    break
        if not isinstance(payload, dict):
            raise ValueError("Payload must be a JSON object.")
        missing = expected_keys.difference(payload.keys())
        if missing:
            raise ValueError(f"Payload missing keys: {', '.join(sorted(missing))}")
        return payload

    def _extract_apply_report(
        self, raw_text: str, artifacts_dir: Path, stage: str
    ) -> Dict | None:
        labels = ["APPLY_REPORT_JSON", "ADDRESSED_ACTIONS_JSON"]
        last_error: ValueError | None = None
        for expected_keys in (
            {"applied_actions", "unapplied_actions"},
            {"applied_actions", "unresolved_actions"},
        ):
            try:
                report = self._extract_wrapped_json_any(raw_text, labels, expected_keys)
                report = self._normalize_apply_report(report)
                write_json(artifacts_dir / f"requirements_apply_report_{stage}.json", report)
                return report
            except ValueError as exc:
                last_error = exc
        try:
            fallback = self._extract_wrapped_json_any(
                raw_text,
                labels,
                {"addressed_actions"},
            )
            report = self._normalize_apply_report(fallback)
            write_json(artifacts_dir / f"requirements_apply_report_{stage}.json", report)
            return report
        except ValueError as exc:
            last_error = exc
        self._record_apply_report_missing(
            artifacts_dir=artifacts_dir,
            stage=stage,
            artifact="requirements",
            raw_text=raw_text,
            reason=str(last_error) if last_error else "Missing apply report.",
        )
        self._requirements_warnings.append(
            {
                "stage": stage,
                "note": "Missing APPLY_REPORT_JSON or ADDRESSED_ACTIONS_JSON.",
                "error": str(last_error) if last_error else "Missing apply report.",
            }
        )
        return None

    def _normalize_apply_report(self, report: Dict) -> Dict:
        normalized = dict(report)
        if "unresolved_actions" in report and "unapplied_actions" not in report:
            normalized["unapplied_actions"] = report.get("unresolved_actions")
        applied_actions = normalized.get("applied_actions")
        if applied_actions is None and "addressed_actions" in report:
            normalized["applied_actions"] = [
                {"action": action, "evidence": ""}
                for action in report.get("addressed_actions", [])
                if isinstance(action, str)
            ]
        elif isinstance(applied_actions, list):
            normalized["applied_actions"] = [
                item if isinstance(item, dict) else {"action": str(item), "evidence": ""}
                for item in applied_actions
                if isinstance(item, (dict, str))
            ]
        return normalized

    def _record_apply_report_missing(
        self,
        artifacts_dir: Path,
        stage: str,
        artifact: str,
        raw_text: str,
        reason: str,
    ) -> None:
        snippet = raw_text.strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        write_json(
            artifacts_dir / "apply_report_missing_warning.json",
            {
                "stage": stage,
                "artifact": artifact,
                "reason": reason,
                "snippet": snippet,
            },
        )

    def _write_enforcement_failed(
        self,
        artifacts_dir: Path,
        stage: str,
        missing_actions: List[str],
        evidence_issues: List[Dict[str, str]],
        raw_text: str,
    ) -> None:
        snippet = raw_text.strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        offending = None
        if evidence_issues:
            offending = {
                "action": evidence_issues[0].get("action"),
                "evidence_snippet": (evidence_issues[0].get("evidence") or "")[:200],
            }
        write_json(
            artifacts_dir / "requirements_enforcement_failed.json",
            {
                "stage": stage,
                "missing_actions": missing_actions,
                "offending_action": offending,
                "snippet": snippet,
            },
        )

    def _validate_apply_report(
        self, report: Dict | None, payload: Dict, required_actions: List[str]
    ) -> tuple[List[str], List[str], List[Dict[str, str]]]:
        errors: List[str] = []
        if not report:
            errors.append("Missing APPLY_REPORT_JSON/ADDRESSED_ACTIONS_JSON.")
            missing_actions = self._missing_required_actions(required_actions, [])
            return errors, missing_actions, []
        report = self._normalize_apply_report(report)
        applied_actions = report.get("applied_actions")
        unapplied_actions = report.get("unapplied_actions")
        if not isinstance(applied_actions, list):
            return [
                "APPLY_REPORT_JSON/ADDRESSED_ACTIONS_JSON.applied_actions must be a list."
            ], list(required_actions), []
        if unapplied_actions not in ([], None):
            if not isinstance(unapplied_actions, list):
                errors.append(
                    "APPLY_REPORT_JSON/ADDRESSED_ACTIONS_JSON.unapplied_actions must be a list."
                )
            elif unapplied_actions:
                errors.append(
                    "APPLY_REPORT_JSON/ADDRESSED_ACTIONS_JSON.unapplied_actions must be empty."
                )
        for entry in applied_actions:
            if not isinstance(entry, dict) or not isinstance(entry.get("action"), str):
                errors.append(
                    "APPLY_REPORT_JSON/ADDRESSED_ACTIONS_JSON.applied_actions entries must include action strings."
                )
                break
        evidence_issues = self._validate_apply_report_evidence(applied_actions, payload)
        if evidence_issues:
            errors.append(
                "APPLY_REPORT_JSON.applied_actions evidence must cite existing requirement IDs."
            )
        missing_actions = self._missing_required_actions(required_actions, applied_actions)
        if missing_actions:
            errors.append(
                "Missing applied_actions entries for required_actions: "
                + "; ".join(missing_actions)
            )
        return errors, missing_actions, evidence_issues

    def _missing_required_actions(
        self, required_actions: List[str], applied_actions: List[Dict] | List[str]
    ) -> List[str]:
        if not required_actions:
            return []
        required = [
            action.strip()
            for action in required_actions
            if isinstance(action, str) and action.strip()
        ]
        if not required:
            return []
        applied_set: set[str] = set()
        if isinstance(applied_actions, list):
            for entry in applied_actions:
                if isinstance(entry, str):
                    if entry.strip():
                        applied_set.add(entry.strip())
                elif isinstance(entry, dict):
                    action = entry.get("action")
                    if isinstance(action, str) and action.strip():
                        applied_set.add(action.strip())
        return [action for action in required if action not in applied_set]

    def _validate_apply_report_evidence(
        self, applied_actions: List[Dict], payload: Dict
    ) -> List[Dict[str, str]]:
        if not isinstance(applied_actions, list):
            return []
        requirement_ids = {
            str(item.get("id"))
            for item in payload.get("requirements", [])
            if isinstance(item, dict) and item.get("id")
        }
        if not requirement_ids:
            return []
        issues: List[Dict[str, str]] = []
        for entry in applied_actions:
            if not isinstance(entry, dict):
                continue
            action = entry.get("action")
            evidence = entry.get("evidence")
            if not isinstance(action, str) or not isinstance(evidence, str):
                continue
            if not any(req_id in evidence for req_id in requirement_ids):
                issues.append({"action": action, "evidence": evidence})
        return issues

    def _remap_apply_report_ids(self, report: Dict | None, id_map: object) -> Dict | None:
        if not isinstance(report, dict):
            return report
        mapping: Dict[str, str] = {}
        if isinstance(id_map, dict):
            mapping = {
                str(old_id): str(new_id)
                for old_id, new_id in id_map.items()
                if isinstance(old_id, str) and isinstance(new_id, str)
            }
        elif isinstance(id_map, list):
            for entry in id_map:
                if isinstance(entry, dict):
                    old_id = entry.get("old") or entry.get("from")
                    new_id = entry.get("new") or entry.get("to")
                    if isinstance(old_id, str) and isinstance(new_id, str):
                        mapping[old_id] = new_id
                elif (
                    isinstance(entry, (list, tuple))
                    and len(entry) == 2
                    and isinstance(entry[0], str)
                    and isinstance(entry[1], str)
                ):
                    mapping[entry[0]] = entry[1]
            if mapping:
                self._requirements_warnings.append(
                    {
                        "stage": "id_map",
                        "note": "Converted list id_map to dict.",
                        "count": len(mapping),
                    }
                )
        if not mapping:
            if id_map is not None:
                self._requirements_warnings.append(
                    {
                        "stage": "id_map",
                        "note": "Unusable id_map; apply report not remapped.",
                        "type": str(type(id_map)),
                    }
                )
            return report
        updated = dict(report)
        applied_actions = updated.get("applied_actions")
        if isinstance(applied_actions, list):
            remapped_actions = []
            for entry in applied_actions:
                if not isinstance(entry, dict):
                    remapped_actions.append(entry)
                    continue
                evidence = entry.get("evidence")
                if isinstance(evidence, str):
                    for old_id, new_id in mapping.items():
                        if old_id in evidence:
                            evidence = evidence.replace(old_id, new_id)
                    entry = dict(entry)
                    entry["evidence"] = evidence
                remapped_actions.append(entry)
            updated["applied_actions"] = remapped_actions
        unapplied_actions = updated.get("unapplied_actions")
        if isinstance(unapplied_actions, list):
            updated["unapplied_actions"] = [
                mapping.get(action, action) if isinstance(action, str) else action
                for action in unapplied_actions
            ]
        return updated

    def _retry_apply_for_actions(
        self,
        brief: str,
        draft: Dict,
        cross_review: Dict,
        final_payload: Dict,
        apply_report: Dict,
        errors: List[str],
        missing_actions: List[str],
        evidence_issues: List[Dict[str, str]],
        adapter: LLMAdapter,
        raw_dir: Path,
        artifacts_dir: Path,
        max_tokens: int,
        expected_keys: set[str],
    ) -> tuple[Dict, Dict]:
        retry_prompt = read_text(self.prompts_dir / "requirements_apply_retry_actions.md")
        retry_payload = {
            "brief": brief,
            "draft": draft,
            "cross_review": cross_review,
            "current": final_payload,
            "apply_report": apply_report,
            "errors": errors,
            "missing_actions": missing_actions,
            "evidence_issues": evidence_issues,
        }
        full_prompt = f"{retry_prompt}\n\nINPUT:\n{json.dumps(retry_payload)}\n"
        write_text(raw_dir / "requirements_apply_retry_actions_prompt.txt", full_prompt)
        response = self._complete(adapter, full_prompt, max_tokens)
        write_text(raw_dir / "requirements_apply_retry_actions_raw.txt", response.raw_text)
        self._write_usage(
            raw_dir / "requirements_apply_retry_actions_usage.json", response
        )
        self._apply_action_retry_used = True
        retry_payload_json = self._extract_wrapped_json(
            response.raw_text,
            "FINAL_REQUIREMENTS_JSON",
            expected_keys,
            context="requirements_apply_retry_actions",
        )
        retry_payload_json, _ = self._repair_artifact_payload(
            "requirements", retry_payload_json, stage="apply_retry_actions"
        )
        report = self._extract_apply_report(
            response.raw_text, artifacts_dir, stage="apply_retry_actions"
        )
        if report is None:
            self._record_apply_report_missing(
                artifacts_dir=artifacts_dir,
                stage="apply_retry_actions",
                artifact="requirements",
                raw_text=response.raw_text,
                reason="Missing APPLY_REPORT_JSON or ADDRESSED_ACTIONS_JSON in apply retry.",
            )
            self._write_enforcement_failed(
                artifacts_dir=artifacts_dir,
                stage="apply_retry_actions",
                missing_actions=missing_actions,
                evidence_issues=evidence_issues,
                raw_text=response.raw_text,
            )
            raise RuntimeError(
                "Apply retry missing action report (APPLY_REPORT_JSON or ADDRESSED_ACTIONS_JSON)."
            )
        (
            report_errors,
            missing_actions,
            evidence_issues,
        ) = self._validate_apply_report(
            report, retry_payload_json, cross_review.get("required_actions", [])
        )
        if report_errors or missing_actions or evidence_issues:
            self._write_enforcement_failed(
                artifacts_dir=artifacts_dir,
                stage="apply_retry_actions",
                missing_actions=missing_actions,
                evidence_issues=evidence_issues,
                raw_text=response.raw_text,
            )
            evidence_snippet = ""
            if evidence_issues:
                evidence_snippet = f" Evidence: {evidence_issues[0].get('evidence', '')[:200]}"
            error_text = "; ".join(report_errors) if report_errors else "missing actions"
            missing_text = ", ".join(missing_actions)
            raise RuntimeError(
                "Apply retry did not satisfy required actions: "
                f"{error_text}. Missing actions: {missing_text}.{evidence_snippet}"
            )
        write_json(
            artifacts_dir / "requirements_apply_retry_actions_extracted.json",
            retry_payload_json,
        )
        write_json(
            artifacts_dir / "requirements_apply_retry_actions_report.json",
            report,
        )
        return retry_payload_json, report

    def _retry_apply_for_final_review(
        self,
        brief: str,
        cross_review: Dict,
        final_payload: Dict,
        final_review: Dict,
        adapter: LLMAdapter,
        raw_dir: Path,
        artifacts_dir: Path,
        max_tokens: int,
    ) -> Dict:
        retry_prompt = read_text(self.prompts_dir / "requirements_apply_final_retry.md")
        retry_payload = {
            "brief": brief,
            "cross_review": cross_review,
            "current": final_payload,
            "final_review": final_review,
        }
        full_prompt = f"{retry_prompt}\n\nINPUT:\n{json.dumps(retry_payload)}\n"
        write_text(raw_dir / "requirements_apply_final_retry_prompt.txt", full_prompt)
        response = self._complete(adapter, full_prompt, max_tokens)
        write_text(raw_dir / "requirements_apply_final_retry_raw.txt", response.raw_text)
        self._write_usage(
            raw_dir / "requirements_apply_final_retry_usage.json", response
        )
        retry_payload_json = self._extract_wrapped_json(
            response.raw_text,
            "FINAL_REQUIREMENTS_JSON",
            {"requirements", "assumptions", "constraints"},
            context="requirements_apply_final_retry",
        )
        retry_payload_json, _ = self._repair_artifact_payload(
            "requirements", retry_payload_json, stage="apply_final_retry"
        )
        report = self._extract_apply_report(
            response.raw_text, artifacts_dir, stage="apply_final_retry"
        )
        if report is not None:
            report_errors, missing_actions, evidence_issues = self._validate_apply_report(
                report, retry_payload_json, cross_review.get("required_actions", [])
            )
            if report_errors or missing_actions or evidence_issues:
                raise RuntimeError(
                    "Final-review retry did not satisfy blocking actions: "
                    + "; ".join(report_errors or ["missing actions"])
                )
        write_json(
            artifacts_dir / "requirements_apply_final_retry_extracted.json",
            retry_payload_json,
        )
        return retry_payload_json

    def _validate_requirements_review(
        self, review: Dict, draft_payload: Dict, limits: RequirementsLimits
    ) -> Dict:
        schema = self._load_schema("requirements_cross_review.schema.json")
        try:
            validate(instance=review, schema=schema)
        except ValidationError as exc:
            self._requirements_warnings.append(
                {"stage": "cross_review", "note": "Review schema invalid.", "error": str(exc)}
            )
            return self._fallback_review(draft_payload, limits)

        gaps = (
            review.get("missing_areas")
            or review.get("weak_requirements")
            or review.get("blocking_issues")
        )
        if gaps and not review.get("required_actions"):
            self._requirements_warnings.append(
                {"stage": "cross_review", "note": "Review missing required_actions; adding fallback."}
            )
            review["required_actions"] = [
                {
                    "id": "A-00",
                    "type": "coverage_gap",
                    "severity": "blocking",
                    "targets": [],
                    "area": None,
                    "instruction": "Add missing requirements for uncovered areas.",
                }
            ]
        return review

    def _fallback_review(self, draft_payload: Dict, limits: RequirementsLimits) -> Dict:
        coverage_counts = self._coverage_counts(draft_payload, limits)
        missing_areas: List[str] = []
        add_count = 0
        if limits.min_per_area is not None:
            for area in limits.coverage_areas:
                current = coverage_counts.get(area, 0)
                target = limits.min_per_area
                add = max(target - current, 0)
                if add:
                    missing_areas.append(area)
                    add_count += add
        return {
            "blocking_issues": [],
            "missing_areas": missing_areas,
            "weak_requirements": [],
            "required_actions": [
                {
                    "id": "A-00",
                    "type": "coverage_gap",
                    "severity": "blocking",
                    "targets": [],
                    "area": None,
                    "instruction": "Add missing requirements for uncovered areas.",
                }
            ],
        }

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
            "final_target_items": limits.final_target_items,
            "add_only_batch_size": limits.add_only_batch_size,
            "add_only_max_rounds": limits.add_only_max_rounds,
            "add_only_min_new_per_area": limits.add_only_min_new_per_area,
            "min_assumptions": limits.assumptions_min,
            "min_constraints": limits.constraints_min,
            "min_student_reqs": limits.min_student_reqs,
            "min_coordinator_reqs": limits.min_coordinator_reqs,
            "min_admin_reqs": limits.min_admin_reqs,
            "min_domain_keyword_hits": limits.min_domain_keyword_hits,
            "domain_keywords": self._DOMAIN_KEYWORDS,
            "coverage_areas": limits.coverage_areas,
            "coverage_keywords": limits.coverage_keywords,
            "min_per_area": limits.min_per_area,
            "coverage_prefix_mode": limits.coverage_prefix_mode,
        }

    def _coverage_keywords_for_area(self, limits: RequirementsLimits, area: str) -> List[str]:
        keywords = limits.coverage_keywords.get(area, [])
        if not keywords and area in self._DEFAULT_COVERAGE_KEYWORDS:
            keywords = self._DEFAULT_COVERAGE_KEYWORDS[area]
        normalized = [value.lower() for value in keywords]
        if area.lower() not in normalized:
            return [area] + keywords
        return keywords

    def _keyword_hits(self, text: str, keywords: List[str]) -> int:
        if not text:
            return 0
        normalized = text.lower()
        hits = 0
        for keyword in keywords:
            keyword_normalized = keyword.lower().strip()
            if not keyword_normalized:
                continue
            if re.search(rf"\b{re.escape(keyword_normalized)}\b", normalized):
                hits += 1
        return hits

    def _missing_coverage_areas(self, payload: Dict, limits: RequirementsLimits) -> List[str]:
        if limits.min_per_area is None or not limits.coverage_areas:
            return []
        min_per_area = limits.min_per_area
        counts = self._coverage_counts(payload, limits)
        missing: List[str] = []
        for area in limits.coverage_areas:
            count = counts.get(area, 0)
            if count < min_per_area:
                missing.append(area)
        return missing

    def _coverage_counts(self, payload: Dict, limits: RequirementsLimits) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        if not limits.coverage_areas:
            return counts
        req_items = payload.get("requirements", [])
        for area in limits.coverage_areas:
            counts[area] = 0
        unmapped = 0
        if limits.coverage_prefix_mode:
            for item in req_items:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()
                match = re.match(r"^\[(.+?)\]\s+", text)
                if not match:
                    unmapped += 1
                    continue
                area = match.group(1).strip()
                if area in counts:
                    counts[area] += 1
                else:
                    unmapped += 1
            if unmapped:
                counts["UNMAPPED"] = unmapped
                if self._coverage_unmapped_count == 0:
                    self._requirements_warnings.append(
                        {
                            "stage": "coverage_prefix",
                            "note": f"{unmapped} requirements missing coverage prefix.",
                        }
                    )
                self._coverage_unmapped_count = unmapped
            return counts
        for area in limits.coverage_areas:
            keywords = self._coverage_keywords_for_area(limits, area)
            counts[area] = sum(
                1
                for item in req_items
                if isinstance(item, dict)
                and self._keyword_hits(str(item.get("text", "")), keywords) > 0
            )
        return counts

    def _filter_additions(
        self, additions: Dict, out_of_scope: List[str]
    ) -> tuple[Dict, List[Dict], List[str]]:
        filtered_out: List[Dict] = []
        reasons: List[str] = []
        filtered_requirements: List[Dict] = []
        forbidden_patterns = [
            r"implement .* measures",
            r"provide a user-friendly interface",
            r"implement .* controls",
        ]
        out_of_scope_terms = [term.lower() for term in out_of_scope if term.strip()]
        for item in additions.get("requirements", []):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            lower_text = text.lower()
            reason = None
            if any(phrase in lower_text for phrase in self._low_quality_phrases()):
                reason = "contains placeholder phrase"
            if any(
                re.search(pattern, lower_text, flags=re.IGNORECASE)
                for pattern in forbidden_patterns
            ):
                reason = "matches filler pattern"
            if not reason and "implement" in lower_text and "measure" in lower_text:
                if any(term.lower() in lower_text for term in self._coverage_area_terms):
                    reason = "generic implement measures phrasing"
            if not reason and any(term in lower_text for term in out_of_scope_terms):
                reason = "references out-of-scope item"
            if reason:
                filtered_out.append(item)
                reasons.append(reason)
            else:
                filtered_requirements.append(item)
        filtered_payload = dict(additions)
        filtered_payload["requirements"] = filtered_requirements
        return filtered_payload, filtered_out, reasons

    def _semantic_fingerprint(self, text: str) -> str:
        normalized = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = [token for token in normalized.split() if len(token) > 2]
        tokens = sorted(set(tokens))
        return " ".join(tokens)

    def _run_add_only_attempt(
        self,
        brief: str,
        limits: RequirementsLimits,
        payload: Dict,
        adapter: LLMAdapter,
        gemini_review: Dict,
        raw_dir: Path,
        artifacts_dir: Path,
        max_tokens: int,
        attempt: int,
        generate_count: int,
        missing_count_before: int | None = None,
        batch_size: int | None = None,
    ) -> tuple[Dict, List[str], Dict[str, object], Dict[str, object]]:
        balance_results = self._balance_check(payload, limits)
        missing_coverage = self._missing_coverage_areas(payload, limits)
        coverage_counts = self._coverage_counts(payload, limits)
        coverage_targets: List[Dict[str, int | str]] = []
        if limits.min_per_area is not None:
            for area in limits.coverage_areas:
                current = coverage_counts.get(area, 0)
                target = limits.min_per_area
                coverage_targets.append(
                    {
                        "area": area,
                        "current": current,
                        "target": target,
                        "add": max(target - current, 0),
                    }
                )
        missing_balance = balance_results.get("missing", {})
        missing_balance_targets = [
            {"target": "student_reqs", "missing": missing_balance.get("student_reqs", 0)},
            {
                "target": "coordinator_reqs",
                "missing": missing_balance.get("coordinator_reqs", 0),
            },
            {"target": "admin_reqs", "missing": missing_balance.get("admin_reqs", 0)},
            {
                "target": "domain_keyword_hits",
                "missing": missing_balance.get("domain_keyword_hits", 0),
            },
        ]
        existing_ids = [
            item.get("id")
            for item in payload.get("requirements", [])
            if isinstance(item, dict)
        ]
        existing_texts = [
            str(item.get("text", ""))[:120]
            for item in payload.get("requirements", [])
            if isinstance(item, dict)
        ]
        existing_fingerprints = [
            self._semantic_fingerprint(str(item.get("text", "")))
            for item in payload.get("requirements", [])
            if isinstance(item, dict)
        ]
        requested_payload = {
            "requested_count": generate_count,
            "batch_size": batch_size,
            "missing_count": missing_count_before,
            "missing_count_before": missing_count_before,
            "missing_areas": missing_coverage,
            "per_area_counts": coverage_counts,
            "effective_tokens": max_tokens,
        }
        write_json(artifacts_dir / f"add_only_round_{attempt}_requested.json", requested_payload)
        retry_prompt = read_text(self.prompts_dir / "requirements_add_only.md")
        retry_payload = {
            "brief": brief,
            "current_requirements": payload,
            "targets": self._requirements_targets_payload(limits),
            "missing_count": max(limits.req_min - len(payload.get("requirements", [])), 0),
            "generate_count": generate_count,
            "missing_coverage_areas": missing_coverage,
            "coverage_counts": coverage_counts,
            "coverage_targets": coverage_targets,
            "min_new_per_area": limits.add_only_min_new_per_area,
            "missing_balance_targets": missing_balance_targets,
            "balance_results": balance_results,
            "out_of_scope": self._out_of_scope_terms,
            "existing_ids": existing_ids,
            "existing_texts": existing_texts,
            "existing_fingerprints": existing_fingerprints,
            "gemini_review": gemini_review,
        }
        missing_balance_lines = [
            f"- {entry['target']}: {entry['missing']}"
            for entry in missing_balance_targets
            if isinstance(entry.get("missing"), int) and entry.get("missing") > 0
        ]
        missing_balance_summary = (
            "\n".join(missing_balance_lines) if missing_balance_lines else "- none"
        )
        existing_ids_summary = ", ".join(str(item) for item in existing_ids if item) or "none"
        existing_texts_summary = "; ".join(text for text in existing_texts if text) or "none"
        missing_coverage_summary = ", ".join(missing_coverage) if missing_coverage else "none"
        coverage_counts_summary = (
            ", ".join(f"{area}={count}" for area, count in coverage_counts.items())
            if coverage_counts
            else "none"
        )
        forbidden_phrases = "; ".join(self._low_quality_phrases())
        out_of_scope_summary = (
            ", ".join(self._out_of_scope_terms) if self._out_of_scope_terms else "none"
        )
        prefix_instruction = ""
        if limits.coverage_prefix_mode:
            areas = ", ".join(limits.coverage_areas)
            prefix_instruction = (
                "\nPrefix each new requirement with [<Coverage Area>] "
                f"using one of: {areas}."
            )
        full_prompt = (
            f"{retry_prompt}\n\nExisting requirement IDs: {existing_ids_summary}\n"
            f"Existing requirement texts (snippets): {existing_texts_summary}\n"
            f"Coverage counts: {coverage_counts_summary}\n"
            f"Missing coverage areas: {missing_coverage_summary}\n"
            f"Missing balance targets:\n{missing_balance_summary}\n"
            f"Out-of-scope items (do NOT include): {out_of_scope_summary}\n"
            f"Generate EXACTLY {generate_count} new requirements.\n"
            "Each new requirement must mention a concrete actor "
            "(Student/Coordinator/Admin/System) and reference a domain object "
            "(procedure/document/deadline/exception/approval/signature/mobility/"
            "notification/audit/integration).\n"
            "Avoid generic wording like \"implement <coverage area> measures\".\n"
            f"Cap 'could' priorities to at most 20% of {generate_count} items.\n"
            f"Forbidden placeholder phrases: {forbidden_phrases}\n\nINPUT:\n"
            f"{json.dumps(retry_payload)}\n"
        )
        if prefix_instruction:
            full_prompt = full_prompt.replace("\n\nINPUT:\n", f"{prefix_instruction}\n\nINPUT:\n")
        additions, merge_report = self._parse_add_only_response(
            attempt=attempt,
            generate_count=generate_count,
            full_prompt=full_prompt,
            retry_payload=retry_payload,
            brief=brief,
            limits=limits,
            payload=payload,
            adapter=adapter,
            raw_dir=raw_dir,
            artifacts_dir=artifacts_dir,
            max_tokens=max_tokens,
        )
        write_json(
            artifacts_dir / f"add_only_round_{attempt}_extracted.json",
            additions,
        )
        write_json(
            artifacts_dir / f"add_only_round_{attempt}_merge_report.json",
            merge_report,
        )
        write_json(artifacts_dir / f"requirements_add_only_retry_{attempt}.json", additions)
        write_json(artifacts_dir / f"add_only_attempt_{attempt}.json", additions)
        write_json(
            artifacts_dir / f"requirements_add_only_attempt_{attempt}.json",
            additions,
        )
        write_json(
            artifacts_dir / f"requirements_add_only_attempt_{attempt}_extracted.json",
            additions,
        )
        additions, _ = self._repair_artifact_payload("requirements", additions, stage="add_only")
        write_json(
            artifacts_dir / f"requirements_add_only_attempt_{attempt}_normalized.json",
            additions,
        )
        additions, filtered, reasons = self._filter_additions(additions, self._out_of_scope_terms)
        attempt_warnings: List[Dict[str, object]] = []
        if filtered:
            self._requirements_filler_filtered.extend(
                [{"item": item, "reason": reason} for item, reason in zip(filtered, reasons)]
            )
            attempt_warnings.extend(
                [
                    {"stage": "filter_additions", "item": item, "reason": reason}
                    for item, reason in zip(filtered, reasons)
                ]
            )
        additions, filtered_out, quality_warnings = self._apply_quality_gate(additions, limits)
        if filtered_out:
            self._requirements_filtered_out.extend(filtered_out)
            attempt_warnings.extend(
                [{"stage": "quality_gate", "item": item} for item in filtered_out]
            )
        if quality_warnings:
            self._requirements_quality_warnings.extend(quality_warnings)
            attempt_warnings.extend(
                [{"stage": "quality_gate", "warning": warning} for warning in quality_warnings]
            )
        payload = self._merge_requirements_additions(
            payload,
            additions,
            dedupe_mode="exact_text",
            ignore_ids=True,
        )
        payload, filtered_out, quality_warnings = self._apply_quality_gate(payload, limits)
        if filtered_out:
            self._requirements_filtered_out.extend(filtered_out)
            attempt_warnings.extend(
                [{"stage": "quality_gate_post_merge", "item": item} for item in filtered_out]
            )
        if quality_warnings:
            self._requirements_quality_warnings.extend(quality_warnings)
            attempt_warnings.extend(
                [{"stage": "quality_gate_post_merge", "warning": warning} for warning in quality_warnings]
            )
        write_json(
            artifacts_dir / f"requirements_after_round_{attempt}.json",
            payload,
        )
        missing_coverage = self._missing_coverage_areas(payload, limits)
        balance_results = self._balance_check(payload, limits)
        self._requirements_balance_results = balance_results
        try:
            schema = self._load_schema("normalized_requirements.schema.json")
            validate(instance=payload, schema=schema)
        except ValidationError as exc:
            self._requirements_warnings.append(
                {"stage": "add_only", "note": "Validation failed after add-only.", "error": str(exc)}
            )
            attempt_warnings.append({"stage": "validation", "error": str(exc)})
        write_json(
            artifacts_dir / f"requirements_add_only_warnings_{attempt}.json",
            {"warnings": attempt_warnings},
        )
        return payload, missing_coverage, balance_results, merge_report

    def _parse_add_only_response(
        self,
        attempt: int,
        generate_count: int,
        full_prompt: str,
        retry_payload: Dict,
        brief: str,
        limits: RequirementsLimits,
        payload: Dict,
        adapter: LLMAdapter,
        raw_dir: Path,
        artifacts_dir: Path,
        max_tokens: int,
    ) -> tuple[Dict, Dict]:
        max_retries = 1
        existing_texts = {
            self._normalize_exact_text(str(item.get("text", "")))
            for item in payload.get("requirements", [])
            if isinstance(item, dict)
        }
        missing_coverage = self._missing_coverage_areas(payload, limits)
        coverage_counts = self._coverage_counts(payload, limits)
        invalid_prefix_samples: List[str] = []
        consecutive_low_count = 0
        consecutive_format_fail = 0
        min_acceptable = max(1, math.ceil(generate_count * 0.5))
        last_report: Dict[str, object] = {}
        for retry_index in range(1, max_retries + 2):
            strict = retry_index > 1
            strict_note = ""
            if strict:
                strict_note = (
                    "\nCOUNT_STRICT: Output exactly generate_count items. "
                    "If you output fewer or more, you fail. No prose."
                )
            if limits.coverage_prefix_mode:
                strict_note += (
                    "\nEvery requirement MUST start with [<Coverage Area>] "
                    "using only the provided coverage_areas."
                )
                if strict:
                    strict_note += "\nPREFIX_STRICT: Invalid prefixes will be rejected."
                if invalid_prefix_samples:
                    sample_list = "; ".join(invalid_prefix_samples[:3])
                    strict_note += f"\nInvalid prefix examples to avoid: {sample_list}"
            requested_payload = {
                "expected_N": generate_count,
                "existing_ids_count": len(
                    [
                        item.get("id")
                        for item in payload.get("requirements", [])
                        if isinstance(item, dict)
                    ]
                ),
                "missing_areas": missing_coverage,
                "coverage_counts": coverage_counts,
            }
            write_json(
                artifacts_dir
                / f"add_only_round_{attempt}_attempt_{retry_index}_requested.json",
                requested_payload,
            )
            prompt = f"{full_prompt}{strict_note}"
            write_text(
                raw_dir / f"add_only_round_{attempt}_attempt_{retry_index}_prompt.txt",
                prompt,
            )
            response = self._complete(adapter, prompt, max_tokens)
            completion_tokens = self._completion_tokens(response)
            if completion_tokens is not None:
                self._add_only_completion_tokens.append(completion_tokens)
            write_text(
                raw_dir / f"add_only_round_{attempt}_attempt_{retry_index}_raw.txt",
                response.raw_text,
            )
            self._write_usage(
                raw_dir / f"add_only_round_{attempt}_attempt_{retry_index}_usage.json",
                response,
            )
            items, shape, warning = self._extract_add_only_items(response.raw_text)
            if warning:
                self._requirements_warnings.append(
                    {
                        "stage": "add_only",
                        "note": warning,
                        "attempt": attempt,
                        "retry": retry_index,
                    }
                )
            write_json(
                artifacts_dir
                / f"add_only_round_{attempt}_attempt_{retry_index}_extracted.json",
                items,
            )
            parsed_count = len(items)
            returned_count = parsed_count
            if returned_count == 0:
                snippet = response.raw_text.strip().replace("\n", " ")
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                raise RuntimeError(
                    "Add-only returned zero items. "
                    f"Expected {generate_count}. effective_max_tokens={max_tokens}. "
                    f"Shape: {shape}. Snippet: {snippet}"
                )
            accepted: List[Dict] = []
            rejected_reasons: List[str] = []
            format_failure = False
            new_texts: set[str] = set()
            valid_prefix_count = 0
            for item in items:
                if not isinstance(item, dict):
                    rejected_reasons.append("invalid_item_type")
                    format_failure = True
                    continue
                text = str(item.get("text", "")).strip()
                if not text:
                    rejected_reasons.append("missing_text")
                    format_failure = True
                    continue
                if limits.coverage_prefix_mode:
                    match = re.match(r"^\[(.+?)\]\s+", text)
                    if not match or match.group(1).strip() not in limits.coverage_areas:
                        rejected_reasons.append("invalid_prefix")
                        if retry_index <= max_retries:
                            format_failure = True
                        invalid_prefix_samples.append(text[:160])
                        continue
                    valid_prefix_count += 1
                normalized_text = self._normalize_exact_text(text)
                if normalized_text in existing_texts or normalized_text in new_texts:
                    rejected_reasons.append("duplicate_text")
                    continue
                new_texts.add(normalized_text)
                accepted.append(item)
            if not limits.coverage_prefix_mode:
                valid_prefix_count = len(accepted)
            accepted_count = len(accepted)
            if returned_count != generate_count:
                rejected_reasons.append("incorrect_count")
            report = {
                "requested_count": generate_count,
                "parsed_count": parsed_count,
                "returned_count": returned_count,
                "valid_prefix_count": valid_prefix_count,
                "accepted_count": accepted_count,
                "rejected_count": len(rejected_reasons),
                "rejected_reasons": rejected_reasons,
                "shape": shape,
                "min_acceptable": min_acceptable,
            }
            last_report = report
            write_json(
                artifacts_dir
                / f"add_only_round_{attempt}_attempt_{retry_index}_merge_report.json",
                report,
            )
            if format_failure:
                consecutive_format_fail += 1
                consecutive_low_count = 0
                if consecutive_format_fail >= 2:
                    snippet = response.raw_text.strip().replace("\n", " ")
                    if len(snippet) > 200:
                        snippet = snippet[:200] + "..."
                    raise RuntimeError(
                        "Add-only format invalid for consecutive retries. "
                        f"Expected {generate_count}. effective_max_tokens={max_tokens}. "
                        f"Shape: {shape}. Snippet: {snippet}"
                    )
                continue
            consecutive_format_fail = 0
            if accepted_count < min_acceptable:
                if returned_count < min_acceptable:
                    consecutive_low_count += 1
                    if consecutive_low_count >= 2:
                        snippet = response.raw_text.strip().replace("\n", " ")
                        if len(snippet) > 200:
                            snippet = snippet[:200] + "..."
                        raise RuntimeError(
                            "Add-only returned too few items for consecutive retries. "
                            f"Expected {generate_count}, got {accepted_count}. "
                            f"effective_max_tokens={max_tokens}. Shape: {shape}. Snippet: {snippet}"
                        )
                else:
                    consecutive_low_count = 0
                continue
            consecutive_low_count = 0
            if not format_failure:
                existing_texts.update(new_texts)
                additions = {"requirements": accepted, "assumptions": [], "constraints": []}
                return additions, report
        if not last_report:
            last_report = {
                "requested_count": generate_count,
                "parsed_count": 0,
                "returned_count": 0,
                "valid_prefix_count": 0,
                "accepted_count": 0,
                "rejected_count": 0,
                "rejected_reasons": ["no_valid_response"],
                "shape": "unknown",
                "min_acceptable": min_acceptable,
            }
        additions = {"requirements": [], "assumptions": [], "constraints": []}
        return additions, last_report

    def _extract_add_only_items(self, raw_text: str) -> tuple[List[Dict], str, str | None]:
        try:
            parsed = extract_json(raw_text)
        except ValueError as exc:
            return [], "parse_failed", f"parse_failed: {exc}"
        if isinstance(parsed, dict):
            for label in ["REQUIREMENTS_ADD_ONLY_JSON", "REQUIREMENTS_JSON"]:
                if label in parsed:
                    wrapper = parsed[label]
                    if isinstance(wrapper, dict):
                        reqs = wrapper.get("requirements", [])
                        if isinstance(reqs, list):
                            return reqs, f"wrapper:{label}", None
                    if isinstance(wrapper, list):
                        return wrapper, f"wrapper:{label}", None
            if "requirements" in parsed and isinstance(parsed.get("requirements"), list):
                return parsed["requirements"], "requirements_object", None
        if isinstance(parsed, list):
            return parsed, "bare_list", None
        snippet = raw_text.strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        return [], "unexpected_type", f"unexpected_type: {snippet}"

    def _format_retry_requirements(
        self,
        brief: str,
        apply_raw: str,
        adapter: LLMAdapter,
        raw_dir: Path,
        artifacts_dir: Path,
        max_tokens: int,
        expected_keys: set[str],
    ) -> Dict:
        retry_prompt = read_text(self.prompts_dir / "requirements_apply_format_retry.md")
        retry_payload = {"brief": brief, "apply_raw": apply_raw}
        full_prompt = f"{retry_prompt}\n\nINPUT:\n{json.dumps(retry_payload)}\n"
        write_text(raw_dir / "requirements_apply_format_retry_prompt.txt", full_prompt)
        response = self._complete(adapter, full_prompt, max_tokens)
        completion_tokens = self._completion_tokens(response)
        if completion_tokens is not None:
            self._format_fix_completion_tokens.append(completion_tokens)
        write_text(raw_dir / "requirements_apply_format_retry_raw.txt", response.raw_text)
        self._write_usage(
            raw_dir / "requirements_apply_format_retry_usage.json", response
        )
        try:
            extracted = self._extract_wrapped_json(
                response.raw_text,
                "FINAL_REQUIREMENTS_JSON",
                expected_keys,
                context="requirements_apply_format_retry",
            )
        except ValueError:
            if self._single_requirement_fallback:
                extracted = self._single_requirement_fallback
            else:
                raise
        write_json(
            artifacts_dir / "requirements_apply_format_retry_extracted.json", extracted
        )
        return extracted

    def _format_fix_requirements(
        self,
        brief: str,
        apply_raw: str,
        adapter: LLMAdapter,
        raw_dir: Path,
        artifacts_dir: Path,
        max_tokens: int,
        expected_keys: set[str],
    ) -> Dict:
        retry_prompt = read_text(self.prompts_dir / "requirements_apply_format_fix.md")
        retry_payload = {"brief": brief, "apply_raw": apply_raw}
        full_prompt = f"{retry_prompt}\n\nINPUT:\n{json.dumps(retry_payload)}\n"
        write_text(raw_dir / "requirements_apply_format_fix_prompt.txt", full_prompt)
        response = self._complete(adapter, full_prompt, max_tokens)
        completion_tokens = self._completion_tokens(response)
        if completion_tokens is not None:
            self._format_fix_completion_tokens.append(completion_tokens)
        write_text(raw_dir / "requirements_apply_format_fix_raw.txt", response.raw_text)
        self._write_usage(
            raw_dir / "requirements_apply_format_fix_usage.json", response
        )
        extracted = self._extract_wrapped_json(
            response.raw_text,
            "FINAL_REQUIREMENTS_JSON",
            expected_keys,
            context="requirements_apply_format_fix",
        )
        return extracted

    def _out_of_scope_from_frontmatter(self, frontmatter: Dict) -> List[str]:
        if not isinstance(frontmatter, dict):
            return []
        out_of_scope = frontmatter.get("out_of_scope", [])
        if isinstance(out_of_scope, str):
            out_of_scope = [out_of_scope]
        if isinstance(out_of_scope, list):
            return [item for item in out_of_scope if isinstance(item, str)]
        return []

    def _coverage_defaults_used(self, frontmatter: Dict) -> bool:
        if not isinstance(frontmatter, dict):
            return True
        coverage_areas = frontmatter.get("coverage_areas")
        if not coverage_areas:
            return True
        if isinstance(coverage_areas, list):
            for entry in coverage_areas:
                if isinstance(entry, dict):
                    keywords = entry.get("keywords", [])
                    if not isinstance(keywords, list) or not keywords:
                        return True
                else:
                    return True
        return False

    def _is_requirement_object(self, candidate: Dict) -> bool:
        keys = set(candidate.keys())
        return {"id", "text", "priority"}.issubset(keys) and "requirements" not in keys

    def _record_parse_repairs(self, context: str | None, repairs: List[str]) -> None:
        if not context or not repairs:
            return
        self._json_parse_repairs.append({"context": context, "repairs": repairs})

    def _record_extraction_debug(
        self,
        context: str | None,
        strategy: str,
        truncation_detected: bool,
        parse_error: str | None,
        debug_enabled: bool,
    ) -> None:
        if not context and not debug_enabled:
            return
        if not context:
            context = "unspecified"
        entry: Dict[str, object] = {
            "context": context,
            "strategy": strategy,
            "truncation_detected": truncation_detected,
        }
        if parse_error:
            entry["parse_error"] = parse_error[:200]
        self._extraction_debug.append(entry)

    def _extract_first_json_object(self, text: str) -> str | None:
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == "\"":
                in_string = not in_string
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return None

    def _raw_decode_json_object(self, text: str) -> Dict | None:
        start = text.find("{")
        if start == -1:
            return None
        decoder = json.JSONDecoder()
        try:
            parsed, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _extract_fenced_json(self, text: str) -> str | None:
        matches = list(re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE))
        if not matches:
            return None
        return matches[0].group(1)

    def _extract_json_after_label(self, text: str, label: str) -> str | None:
        match = re.search(re.escape(label), text, re.IGNORECASE)
        if not match:
            return None
        snippet = text[match.end():]
        return self._extract_first_json_object(snippet)

    def _extract_json_after_marker(self, text: str, label: str) -> str | None:
        match = re.search(rf"{re.escape(label)}\s*:\s*", text, re.IGNORECASE)
        if not match:
            return None
        snippet = text[match.end():]
        return self._extract_first_json_object(snippet)

    def _extract_between_braces(self, text: str) -> str | None:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1]

    def _largest_balanced_json(self, text: str) -> str | None:
        objects: List[str] = []
        idx = 0
        while idx < len(text):
            candidate = self._extract_first_json_object(text[idx:])
            if not candidate:
                break
            objects.append(candidate)
            idx += text[idx:].find(candidate) + len(candidate)
        if not objects:
            return None
        return max(objects, key=len)

    def _detect_truncation(self, text: str) -> bool:
        stripped = text.rstrip()
        if stripped and not stripped.endswith("}"):
            return True
        depth_brace = 0
        depth_bracket = 0
        in_string = False
        escape = False
        for ch in text:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == "\"":
                in_string = not in_string
            if in_string:
                continue
            if ch == "{":
                depth_brace += 1
            elif ch == "}":
                depth_brace -= 1
            elif ch == "[":
                depth_bracket += 1
            elif ch == "]":
                depth_bracket -= 1
        return depth_brace != 0 or depth_bracket != 0

    def _low_quality_phrases(self) -> List[str]:
        return [
            "as described in the brief",
            "define and enforce behavior",
            "user-friendly interface",
            "provide guidelines",
        ]

    def _low_quality_reason(self, text: str) -> str | None:
        if not text:
            return "empty requirement text"
        normalized = text.lower().strip()
        for phrase in self._low_quality_phrases():
            if phrase in normalized:
                return f"contains placeholder phrase '{phrase}'"
        if self._coverage_area_terms:
            generic_verbs = [
                "support",
                "handle",
                "cover",
                "address",
                "define",
                "provide",
                "ensure",
                "include",
                "manage",
            ]
            domain_verbs = [
                "create",
                "submit",
                "approve",
                "reject",
                "review",
                "upload",
                "download",
                "notify",
                "schedule",
                "assign",
                "track",
                "validate",
                "sign",
                "archive",
                "escalate",
                "complete",
                "resubmit",
            ]
            has_domain_verb = any(
                re.search(rf"\b{verb}\b", normalized) for verb in domain_verbs
            )
            for area in self._coverage_area_terms:
                area_normalized = area.lower().strip()
                if not area_normalized:
                    continue
                if area_normalized in normalized and not has_domain_verb:
                    for verb in generic_verbs:
                        if re.search(rf"\b{verb}\b", normalized):
                            return "restate of coverage area without domain action"
        return None

    def is_low_quality_requirement(self, text: str) -> bool:
        return self._low_quality_reason(text) is not None

    def _apply_quality_gate(
        self, payload: Dict, limits: RequirementsLimits, remove_items: bool = True
    ) -> tuple[Dict, List[Dict], List[Dict]]:
        filtered_out: List[Dict] = []
        warnings: List[Dict] = []
        filtered_requirements: List[Dict] = []
        for item in payload.get("requirements", []):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            reason = self._low_quality_reason(text)
            if reason:
                filtered_out.append(item)
                warnings.append(
                    {"id": item.get("id"), "text": text, "reason": reason}
                )
            else:
                filtered_requirements.append(item)
        filtered_payload = dict(payload)
        filtered_payload["requirements"] = filtered_requirements if remove_items else payload.get(
            "requirements", []
        )
        return filtered_payload, filtered_out, warnings

    def _balance_check(self, payload: Dict, limits: RequirementsLimits) -> Dict[str, object]:
        role_patterns = {
            "student": ["student", "learner"],
            "coordinator": ["coordinator", "advisor"],
            "admin": ["administrator", "admin"],
        }
        role_counts = {key: 0 for key in role_patterns}
        keyword_hits = 0
        for item in payload.get("requirements", []):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).lower()
            for role, terms in role_patterns.items():
                if any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms):
                    role_counts[role] += 1
            for keyword in self._DOMAIN_KEYWORDS:
                keyword_hits += len(re.findall(rf"\b{re.escape(keyword)}\b", text))
        missing = {
            "student_reqs": max(limits.min_student_reqs - role_counts["student"], 0),
            "coordinator_reqs": max(
                limits.min_coordinator_reqs - role_counts["coordinator"], 0
            ),
            "admin_reqs": max(limits.min_admin_reqs - role_counts["admin"], 0),
            "domain_keyword_hits": max(
                limits.min_domain_keyword_hits - keyword_hits, 0
            ),
        }
        meets = all(value <= 0 for value in missing.values())
        return {
            "counts": {**role_counts, "domain_keyword_hits": keyword_hits},
            "targets": {
                "student_reqs": limits.min_student_reqs,
                "coordinator_reqs": limits.min_coordinator_reqs,
                "admin_reqs": limits.min_admin_reqs,
                "domain_keyword_hits": limits.min_domain_keyword_hits,
            },
            "missing": missing,
            "meets": meets,
        }

    def _gemini_missing_points(self, review: Dict) -> List[str]:
        points: List[str] = []
        for entry in review.get("missing_areas", []):
            if isinstance(entry, str) and entry.strip():
                points.append(entry.strip())
        for entry in review.get("blocking_issues", []):
            if isinstance(entry, str) and entry.strip():
                points.append(f"Blocking issue: {entry.strip()}")
        for entry in review.get("weak_requirements", []):
            if isinstance(entry, str) and entry.strip():
                points.append(f"Weak requirement: {entry.strip()}")
        for action in review.get("required_actions", []):
            if not isinstance(action, dict):
                continue
            action_id = action.get("id")
            instruction = action.get("instruction")
            if action_id and instruction:
                points.append(f"{action_id}: {instruction}")
        seen = set()
        deduped = []
        for point in points:
            if point not in seen:
                seen.add(point)
                deduped.append(point)
        return deduped

    def _add_only_requirements_loop(
        self,
        brief: str,
        limits: RequirementsLimits,
        payload: Dict,
        adapter: LLMAdapter,
        gemini_review: Dict,
        raw_dir: Path,
        artifacts_dir: Path,
        max_tokens: int,
    ) -> tuple[Dict, List[str], int, Dict[str, object], List[int], List[Dict[str, object]]]:
        attempts = 0
        requested_counts: List[int] = []
        round_counts: List[Dict[str, object]] = []
        balance_results = self._balance_check(payload, limits)
        missing_coverage = self._missing_coverage_areas(payload, limits)
        start_count = len(payload.get("requirements", []))
        start_missing = max(limits.req_min - start_count, 0)
        if start_missing <= 0:
            return payload, missing_coverage, attempts, balance_results, requested_counts, round_counts
        max_attempts = 10
        while attempts < max_attempts:
            current_count = len(payload.get("requirements", []))
            missing_n = max(limits.req_min - current_count, 0)
            if missing_n <= 0:
                return payload, missing_coverage, attempts, balance_results, requested_counts, round_counts
            batch_size = min(25, missing_n)
            self._add_only_batch_size_used = batch_size
            generate_count = batch_size
            attempts += 1
            requested_counts.append(generate_count)
            before_count = current_count
            (
                payload,
                missing_coverage,
                balance_results,
                merge_report,
            ) = self._run_add_only_attempt(
                brief=brief,
                limits=limits,
                payload=payload,
                adapter=adapter,
                gemini_review=gemini_review,
                raw_dir=raw_dir,
                artifacts_dir=artifacts_dir,
                max_tokens=max_tokens,
                attempt=attempts,
                generate_count=generate_count,
                missing_count_before=missing_n,
                batch_size=batch_size,
            )
            after_count = len(payload.get("requirements", []))
            round_counts.append(
                {
                    "round": attempts,
                    "before_count": before_count,
                    "after_count": after_count,
                    "missing_before": max(limits.req_min - before_count, 0),
                    "missing_after": max(limits.req_min - after_count, 0),
                    "requested_count": generate_count,
                    "parsed_count": merge_report.get("parsed_count"),
                    "accepted_count": merge_report.get("accepted_count"),
                    "rejected_count": merge_report.get("rejected_count"),
                }
            )
        current_count = len(payload.get("requirements", []))
        missing_n = max(limits.req_min - current_count, 0)
        if missing_n > 0:
            self._requirements_warnings.append(
                {
                    "stage": "add_only",
                    "note": "Targets unmet after add-only attempts.",
                    "missing_count": missing_n,
                    "missing_coverage": missing_coverage,
                    "balance_results": balance_results,
                    "max_rounds": max_attempts,
                }
            )
            raise RuntimeError(
                "Requirements minimum unmet after add-only rounds. "
                f"Missing {missing_n} after {max_attempts} rounds."
            )
        return payload, missing_coverage, attempts, balance_results, requested_counts, round_counts

    def _run_final_review_add_only(
        self,
        brief: str,
        limits: RequirementsLimits,
        payload: Dict,
        adapter: LLMAdapter,
        gemini_adapter: LLMAdapter,
        cross_review: Dict,
        raw_dir: Path,
        artifacts_dir: Path,
        max_tokens: int,
        attempt_offset: int,
    ) -> tuple[Dict, bool]:
        prompt = read_text(self.prompts_dir / "requirements_gemini_final_review.md")
        review_payload = {
            "brief": brief,
            "requirements": payload,
            "targets": self._requirements_targets_payload(limits),
        }
        full_prompt = f"{prompt}\n\nINPUT:\n{json.dumps(review_payload)}\n"
        write_text(raw_dir / "requirements_final_review_prompt.txt", full_prompt)
        response = self._complete(gemini_adapter, full_prompt, max_tokens)
        write_text(raw_dir / "requirements_final_review_raw.txt", response.raw_text)
        self._write_usage(raw_dir / "requirements_final_review_usage.json", response)
        extracted = self._safe_extract_json(response.raw_text)
        write_json(artifacts_dir / "requirements_final_review_extracted.json", extracted)
        normalized = self._normalize_final_review(extracted)
        write_json(artifacts_dir / "requirements_final_review_normalized.json", normalized)
        self._gemini_final_review_used = True

        if normalized.get("ok") is False:
            payload = self._retry_apply_for_final_review(
                brief=brief,
                cross_review=cross_review,
                final_payload=payload,
                final_review=normalized,
                adapter=adapter,
                raw_dir=raw_dir,
                artifacts_dir=artifacts_dir,
                max_tokens=max_tokens,
            )
            self._final_review_retry_used = True

        missing_n = max(limits.req_min - len(payload.get("requirements", [])), 0)
        missing_areas = normalized.get("missing_areas", [])
        if not missing_n and not missing_areas:
            return payload, False
        generate_count = missing_n if missing_n > 0 else max(1, len(missing_areas))
        if limits.req_max is not None:
            remaining = max(limits.req_max - len(payload.get("requirements", [])), 0)
            generate_count = min(generate_count, remaining)
        generate_count = min(generate_count, 12)
        if generate_count <= 0:
            self._requirements_warnings.append(
                {
                    "stage": "final_review_add_only",
                    "note": "Skipping add-only due to max requirement limit.",
                }
            )
            return payload, False
        payload, _, _, _ = self._run_add_only_attempt(
            brief=brief,
            limits=limits,
            payload=payload,
            adapter=adapter,
            gemini_review=normalized,
            raw_dir=raw_dir,
            artifacts_dir=artifacts_dir,
            max_tokens=max_tokens,
            attempt=attempt_offset + 1,
            generate_count=generate_count,
        )
        return payload, True

    def _fallback_requirements(self, missing_count: int, limits: RequirementsLimits) -> List[Dict]:
        placeholders: List[Dict] = []
        areas = limits.coverage_areas or ["General"]
        for index in range(missing_count):
            area = areas[index % len(areas)]
            placeholders.append(
                {
                    "id": None,
                    "text": f"The system shall define and enforce behavior for {area} as described in the brief.",
                    "priority": "should",
                }
            )
        return placeholders

    def _normalize_final_review(self, review: Dict) -> Dict[str, object]:
        ok = review.get("ok")
        if isinstance(ok, (bool, int)):
            ok_value = bool(ok)
        else:
            ok_value = True
            self._requirements_warnings.append(
                {"stage": "final_review", "note": "Missing ok flag; defaulted to true."}
            )
        def normalize_list(value: object) -> List[str]:
            if not isinstance(value, list):
                return []
            return [str(item).strip() for item in value if str(item).strip()]

        return {
            "ok": ok_value,
            "missing_areas": normalize_list(review.get("missing_areas")),
            "duplicate_candidates": normalize_list(review.get("duplicate_candidates")),
            "top_ambiguities": normalize_list(review.get("top_ambiguities")),
        }

    def _add_assumptions_constraints(
        self,
        brief: str,
        limits: RequirementsLimits,
        payload: Dict,
        adapter: LLMAdapter,
        raw_dir: Path,
        artifacts_dir: Path,
        max_tokens: int,
    ) -> tuple[Dict, int, int]:
        current_assumptions = payload.get("assumptions", [])
        current_constraints = payload.get("constraints", [])
        missing_assumptions = max(limits.assumptions_min - len(current_assumptions), 0)
        missing_constraints = max(limits.constraints_min - len(current_constraints), 0)
        if missing_assumptions <= 0 and missing_constraints <= 0:
            return payload, 0, 0
        prompt = read_text(
            self.prompts_dir / "requirements_assumptions_constraints_micro.md"
        )
        payload_input = {
            "brief": brief,
            "missing_assumptions": missing_assumptions,
            "missing_constraints": missing_constraints,
            "existing_assumptions": current_assumptions,
            "existing_constraints": current_constraints,
        }
        full_prompt = f"{prompt}\nINPUT:\n{json.dumps(payload_input)}\n"
        write_text(raw_dir / "assumptions_constraints_add_only_prompt.txt", full_prompt)
        response = self._complete(adapter, full_prompt, max_tokens)
        write_text(raw_dir / "assumptions_constraints_add_only_raw.txt", response.raw_text)
        self._write_usage(
            raw_dir / "assumptions_constraints_add_only_usage.json", response
        )
        try:
            additions = self._extract_wrapped_json_any(
                response.raw_text,
                ["REQUIREMENTS_JSON"],
                {"requirements", "assumptions", "constraints"},
            )
        except ValueError as exc:
            self._requirements_warnings.append(
                {
                    "stage": "assumptions_constraints_add_only",
                    "note": "Assumptions/constraints add-only extraction failed.",
                    "error": str(exc),
                }
            )
            return payload, 0, 0
        write_json(
            artifacts_dir / "assumptions_constraints_add_only.json", additions
        )
        additions, _ = self._repair_artifact_payload(
            "requirements", additions, stage="assumptions_constraints_add_only"
        )
        updated = self._merge_requirements_additions(payload, additions)
        assumptions_added = max(len(updated.get("assumptions", [])) - len(current_assumptions), 0)
        constraints_added = max(len(updated.get("constraints", [])) - len(current_constraints), 0)
        return updated, assumptions_added, constraints_added

    def _expand_generic_requirements(
        self,
        brief: str,
        limits: RequirementsLimits,
        payload: Dict,
        adapter: LLMAdapter,
        raw_dir: Path,
        artifacts_dir: Path,
        max_tokens: int,
    ) -> tuple[Dict, int]:
        generic_items = self._generic_requirements(payload, limits)
        req_count = len(payload.get("requirements", []))
        if not req_count:
            return payload, 0
        generic_ratio = len(generic_items) / req_count
        if len(generic_items) <= 10 and generic_ratio <= 0.25:
            return payload, 0
        retry_prompt = read_text(self.prompts_dir / "requirements_expand_generic.md")
        retry_payload = {
            "brief": brief,
            "generic_requirements": generic_items,
        }
        full_prompt = f"{retry_prompt}\n\nINPUT:\n{json.dumps(retry_payload)}\n"
        write_text(raw_dir / "requirements_expand_generic_attempt1_prompt.txt", full_prompt)
        response = self._complete(adapter, full_prompt, max_tokens)
        write_text(
            raw_dir / "requirements_expand_generic_attempt1_raw.txt",
            response.raw_text,
        )
        self._write_usage(
            raw_dir / "requirements_expand_generic_attempt1_usage.json", response
        )
        try:
            replacements = self._extract_wrapped_json_any(
                response.raw_text,
                ["REPLACEMENTS_JSON"],
                {"replacements"},
            )
        except ValueError as exc:
            self._requirements_warnings.append(
                {
                    "stage": "expand_generic",
                    "note": "Expand-generic extraction failed.",
                    "error": str(exc),
                }
            )
            return payload, 0
        payload = self._apply_requirement_replacements(payload, replacements)
        payload, _ = self._repair_artifact_payload("requirements", payload, stage="expand_generic")
        write_json(
            artifacts_dir / "requirements_expand_generic_attempt1.json",
            {"generic": generic_items, "replacements": replacements, "merged": payload},
        )
        try:
            schema = self._load_schema("normalized_requirements.schema.json")
            validate(instance=payload, schema=schema)
        except ValidationError as exc:
            self._requirements_warnings.append(
                {
                    "stage": "expand_generic",
                    "note": "Validation failed after expand-generic.",
                    "error": str(exc),
                }
            )
        return payload, 1

    def _generic_requirements(
        self, payload: Dict, limits: RequirementsLimits
    ) -> List[Dict[str, str]]:
        patterns = [
            r"provide .* features",
            r"provide .* capabilities",
            r"implement .* functionalities",
            r"implement .* features",
        ]
        generic_items: List[Dict[str, str]] = []
        for item in payload.get("requirements", []):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", ""))
            lower_text = text.lower()
            is_generic = False
            for area in limits.coverage_areas:
                if area in text:
                    is_generic = True
                    break
            if not is_generic:
                for pattern in patterns:
                    if re.search(pattern, lower_text):
                        is_generic = True
                        break
            if is_generic:
                req_id = str(item.get("id", ""))
                generic_items.append({"id": req_id, "text": text})
        return generic_items

    def _apply_requirement_replacements(self, payload: Dict, replacements: Dict) -> Dict:
        replacement_map: Dict[str, List[Dict]] = {}
        for entry in replacements.get("replacements", []):
            if not isinstance(entry, dict):
                continue
            from_id = entry.get("from")
            into = entry.get("into", [])
            if isinstance(from_id, str) and isinstance(into, list):
                replacement_map[from_id] = [item for item in into if isinstance(item, dict)]

        if not replacement_map:
            return payload

        updated: List[Dict] = []
        for item in payload.get("requirements", []):
            if not isinstance(item, dict):
                continue
            req_id = item.get("id")
            if isinstance(req_id, str) and req_id in replacement_map:
                updated.extend(replacement_map[req_id])
            else:
                updated.append(item)
        payload["requirements"] = updated
        return payload

    def _merge_requirements_additions(
        self,
        base: Dict,
        additions: Dict,
        dedupe_mode: str = "full",
        ignore_ids: bool = False,
    ) -> Dict:
        merged = {
            "requirements": list(base.get("requirements", [])),
            "assumptions": list(base.get("assumptions", [])),
            "constraints": list(base.get("constraints", [])),
        }
        existing_ids = {
            str(item.get("id")).strip()
            for item in merged["requirements"]
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        existing_texts = {
            str(item.get("text", "")).strip().lower()
            for item in merged["requirements"]
            if isinstance(item, dict)
        }
        existing_fingerprints = set()
        if dedupe_mode == "full":
            existing_fingerprints = {
                self._semantic_fingerprint(str(item.get("text", "")).strip())
                for item in merged["requirements"]
                if isinstance(item, dict)
            }
        for item in additions.get("requirements", []):
            if isinstance(item, dict):
                item_id = item.get("id")
                item_text = str(item.get("text", "")).strip()
                normalized_text = item_text.lower()
                fingerprint = self._semantic_fingerprint(item_text)
                normalized_exact = self._normalize_exact_text(item_text)
                if not ignore_ids and isinstance(item_id, str) and item_id in existing_ids:
                    self._requirements_duplicates_debug.append(
                        {
                            "reason": "duplicate_id",
                            "id": item_id,
                            "text": item_text,
                        }
                    )
                    continue
                if normalized_exact and normalized_exact in existing_texts:
                    self._requirements_duplicates_debug.append(
                        {
                            "reason": "duplicate_text",
                            "id": item_id,
                            "text": item_text,
                        }
                    )
                    continue
                if dedupe_mode == "full" and fingerprint and fingerprint in existing_fingerprints:
                    self._requirements_duplicates_debug.append(
                        {
                            "reason": "duplicate_semantic",
                            "id": item_id,
                            "text": item_text,
                        }
                    )
                    continue
                if isinstance(item_id, str) and not ignore_ids:
                    existing_ids.add(item_id)
                if normalized_exact:
                    existing_texts.add(normalized_exact)
                if dedupe_mode == "full" and fingerprint:
                    existing_fingerprints.add(fingerprint)
                if ignore_ids:
                    item = dict(item)
                    item["id"] = None
                merged["requirements"].append(item)
        for item in additions.get("assumptions", []):
            if isinstance(item, str):
                merged["assumptions"].append(item)
        for item in additions.get("constraints", []):
            if isinstance(item, str):
                merged["constraints"].append(item)
        return merged

    def _normalize_exact_text(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        return normalized

    def _normalize_requirement_ids(
        self, payload: Dict, changelog: Dict | None = None
    ) -> tuple[Dict, bool, Dict[str, str], Dict | None]:
        if not isinstance(payload, dict):
            return payload, False, {}, changelog
        items = payload.get("requirements", [])
        if not isinstance(items, list):
            return payload, False, {}, changelog
        normalized = False
        id_map: Dict[str, str] = {}
        normalized_items: List[Dict] = []
        sequence = 1
        for item in items:
            if not isinstance(item, dict):
                continue
            current_id = item.get("id")
            new_id = f"REQ-{sequence:03d}"
            sequence += 1
            if current_id != new_id:
                id_map[str(current_id)] = new_id
                normalized = True
            item["id"] = new_id
            normalized_items.append(item)
        payload["requirements"] = normalized_items
        if id_map and changelog is not None:
            changelog = self._rewrite_changelog_ids(
                changelog,
                [{"from": old_id, "to": new_id} for old_id, new_id in id_map.items()],
            )
        return payload, normalized, id_map, changelog

    def _rewrite_changelog_ids(self, changelog: Dict, id_map: List[Dict[str, str]]) -> Dict:
        mapping = {entry["from"]: entry["to"] for entry in id_map if "from" in entry and "to" in entry}

        def remap(value: str) -> str:
            return mapping.get(value, value)

        def remap_list(values: List[str]) -> List[str]:
            return [remap(value) for value in values if isinstance(value, str)]

        updated = dict(changelog)
        for key in ["added", "removed", "replacements"]:
            if isinstance(updated.get(key), list):
                updated[key] = remap_list(updated[key])
        if isinstance(updated.get("splits"), list):
            splits = []
            for entry in updated["splits"]:
                if isinstance(entry, dict) and "from" in entry and "into" in entry:
                    splits.append(
                        {
                            "from": remap(str(entry.get("from"))),
                            "into": remap_list(entry.get("into", [])),
                        }
                    )
                else:
                    splits.append(entry)
            updated["splits"] = splits
        return updated

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
            final_target_items = summary.get("final_target_items")
            initial_count = summary.get("initial_count")
            actual_count = summary.get("actual_count")
            assumptions_count = summary.get("assumptions_count")
            constraints_count = summary.get("constraints_count")
            lead_budget_tokens = summary.get("lead_budget_max_output_tokens")
            apply_budget_tokens = summary.get("apply_budget_max_output_tokens")
            lead_effective_tokens = summary.get("lead_effective_max_output_tokens")
            apply_effective_tokens = summary.get("apply_effective_max_output_tokens")
            lead_max_tokens = summary.get("lead_max_output_tokens")
            apply_max_tokens = summary.get("apply_max_output_tokens")
            cli_max_tokens = summary.get("cli_max_output_tokens")
            add_only_max_tokens = summary.get("add_only_max_output_tokens")
            format_retry_max_tokens = summary.get("format_retry_max_output_tokens")
            effective_add_only_tokens = summary.get("effective_add_only_max_output_tokens")
            effective_format_fix_tokens = summary.get("effective_format_fix_max_output_tokens")
            lead_completion_tokens = summary.get("lead_completion_tokens")
            apply_completion_tokens = summary.get("apply_completion_tokens")
            add_only_completion_tokens = summary.get("add_only_completion_tokens")
            format_fix_completion_tokens = summary.get("format_fix_completion_tokens")
            assumptions_constraints_max_tokens = summary.get(
                "assumptions_constraints_max_output_tokens"
            )
            missing_coverage = summary.get("missing_coverage_areas", [])
            add_only_attempts = summary.get("add_only_attempts")
            total_add_only_attempts = summary.get("total_add_only_attempts")
            missing_before_add_only = summary.get("missing_before_add_only")
            missing_after_add_only = summary.get("missing_after_add_only")
            count_before_add_only = summary.get("count_before_add_only")
            count_after_add_only = summary.get("count_after_add_only")
            add_only_chunk_size = summary.get("add_only_chunk_size")
            add_only_requested = summary.get("add_only_requested", [])
            add_only_parse_failures = summary.get("add_only_parse_failures")
            add_only_round_counts = summary.get("add_only_round_counts")
            expand_generic_attempts = summary.get("expand_generic_attempts")
            id_normalized = summary.get("id_normalized")
            review_actions_applied = summary.get("review_actions_applied")
            shape_normalized = summary.get("requirements_shape_normalized")
            filtered_out_count = summary.get("filtered_out_count")
            filler_filtered_count = summary.get("filler_filtered_count")
            dedupe_count = summary.get("dedupe_count")
            balance_results = summary.get("balance_check_results", {})
            coverage_fix_used = summary.get("coverage_fix_used")
            assumptions_added = summary.get("assumptions_added")
            constraints_added = summary.get("constraints_added")
            assumptions_fixed = summary.get("assumptions_fixed")
            constraints_fixed = summary.get("constraints_fixed")
            apply_format_retry_used = summary.get("apply_format_retry_used")
            coverage_counts = summary.get("coverage_counts", {})
            wrapper_repairs_applied = summary.get("wrapper_repairs_applied")
            gemini_review_present = summary.get("gemini_review_present")
            gemini_review_used = summary.get("gemini_review_used")
            gemini_final_review_used = summary.get("gemini_final_review_used")
            post_review_add_only_used = summary.get("post_review_add_only_used")
            min_enforcement_unmet = summary.get("min_enforcement_unmet")
            final_review_retry_used = summary.get("final_review_retry_used")
            coverage_unmapped_count = summary.get("coverage_unmapped_count")
            apply_action_retry_used = summary.get("apply_action_retry_used")
            gemini_cross_review_error = summary.get("gemini_cross_review_error")
            gemini_cross_review_skipped = summary.get("gemini_cross_review_skipped")
            gemini_selected_model = summary.get("gemini_selected_model")
            gemini_error_summary = summary.get("gemini_error_summary")
            cross_review_parse_error = summary.get("cross_review_parse_error")
            lines.append(f"- target_min_items: {target_min_items}")
            if final_target_items is not None:
                lines.append(f"- final_target_items: {final_target_items}")
            lines.append(f"- initial_count: {initial_count}")
            lines.append(f"- final_count: {actual_count}")
            if lead_budget_tokens is not None:
                lines.append(f"- lead_budget_max_output_tokens: {lead_budget_tokens}")
            if apply_budget_tokens is not None:
                lines.append(f"- apply_budget_max_output_tokens: {apply_budget_tokens}")
            if lead_effective_tokens is not None:
                lines.append(f"- lead_effective_max_output_tokens: {lead_effective_tokens}")
            if apply_effective_tokens is not None:
                lines.append(f"- apply_effective_max_output_tokens: {apply_effective_tokens}")
            if lead_max_tokens is not None:
                lines.append(f"- lead_max_output_tokens: {lead_max_tokens}")
            if apply_max_tokens is not None:
                lines.append(f"- apply_max_output_tokens: {apply_max_tokens}")
            if cli_max_tokens is not None:
                lines.append(f"- cli_max_output_tokens: {cli_max_tokens}")
            if add_only_max_tokens is not None:
                lines.append(f"- add_only_max_output_tokens: {add_only_max_tokens}")
            if format_retry_max_tokens is not None:
                lines.append(f"- format_retry_max_output_tokens: {format_retry_max_tokens}")
            if effective_add_only_tokens is not None:
                lines.append(f"- effective_add_only_max_output_tokens: {effective_add_only_tokens}")
            if effective_format_fix_tokens is not None:
                lines.append(
                    f"- effective_format_fix_max_output_tokens: {effective_format_fix_tokens}"
                )
            if lead_completion_tokens is not None:
                lines.append(f"- lead_completion_tokens: {lead_completion_tokens}")
            if apply_completion_tokens is not None:
                lines.append(f"- apply_completion_tokens: {apply_completion_tokens}")
            if isinstance(add_only_completion_tokens, list) and add_only_completion_tokens:
                tokens_str = ", ".join(str(token) for token in add_only_completion_tokens)
                lines.append(f"- add_only_completion_tokens: {tokens_str}")
            if isinstance(format_fix_completion_tokens, list) and format_fix_completion_tokens:
                tokens_str = ", ".join(str(token) for token in format_fix_completion_tokens)
                lines.append(f"- format_fix_completion_tokens: {tokens_str}")
            if assumptions_constraints_max_tokens is not None:
                lines.append(
                    f"- assumptions_constraints_max_output_tokens: {assumptions_constraints_max_tokens}"
                )
            if assumptions_count is not None:
                lines.append(f"- assumptions_count: {assumptions_count}")
            if constraints_count is not None:
                lines.append(f"- constraints_count: {constraints_count}")
            if missing_coverage:
                lines.append(f"- missing_coverage_areas: {', '.join(missing_coverage)}")
            else:
                lines.append("- missing_coverage_areas: none")
            if isinstance(coverage_counts, dict) and coverage_counts:
                counts_str = ", ".join(
                    f"{area}={count}" for area, count in coverage_counts.items()
                )
                lines.append(f"- coverage_counts: {counts_str}")
            lines.append(f"- add_only_attempts: {add_only_attempts}")
            if total_add_only_attempts is not None:
                lines.append(f"- total_add_only_attempts: {total_add_only_attempts}")
            if missing_before_add_only is not None:
                lines.append(f"- missing_before_add_only: {missing_before_add_only}")
            if missing_after_add_only is not None:
                lines.append(f"- missing_after_add_only: {missing_after_add_only}")
            if count_before_add_only is not None:
                lines.append(f"- count_before_add_only: {count_before_add_only}")
            if count_after_add_only is not None:
                lines.append(f"- count_after_add_only: {count_after_add_only}")
            if add_only_chunk_size is not None:
                lines.append(f"- add_only_chunk_size: {add_only_chunk_size}")
            if isinstance(add_only_requested, list) and add_only_requested:
                lines.append(
                    "- add_only_requested: " + ", ".join(str(value) for value in add_only_requested)
                )
            if add_only_parse_failures is not None:
                lines.append(f"- add_only_parse_failures: {add_only_parse_failures}")
            if isinstance(add_only_round_counts, list) and add_only_round_counts:
                lines.append("- add_only_round_counts:")
                for entry in add_only_round_counts:
                    if not isinstance(entry, dict):
                        continue
                    round_id = entry.get("round")
                    before_count = entry.get("before_count")
                    after_count = entry.get("after_count")
                    missing_before = entry.get("missing_before")
                    missing_after = entry.get("missing_after")
                    requested_count = entry.get("requested_count")
                    parsed_count = entry.get("parsed_count")
                    accepted_count = entry.get("accepted_count")
                    rejected_count = entry.get("rejected_count")
                    lines.append(
                        "  - "
                        f"round {round_id}: before={before_count} after={after_count} "
                        f"missing_before={missing_before} missing_after={missing_after} "
                        f"requested={requested_count} parsed={parsed_count} "
                        f"accepted={accepted_count} rejected={rejected_count}"
                    )
            lines.append(f"- expand_generic_attempts: {expand_generic_attempts}")
            lines.append(f"- id_normalized: {'yes' if id_normalized else 'no'}")
            lines.append(f"- review_actions_applied: {'yes' if review_actions_applied else 'no'}")
            lines.append(
                f"- requirements_shape_normalized: {'yes' if shape_normalized else 'no'}"
            )
            if filtered_out_count is not None:
                lines.append(f"- filtered_out_count: {filtered_out_count}")
            if filler_filtered_count is not None:
                lines.append(f"- filler_filtered_count: {filler_filtered_count}")
            if dedupe_count is not None:
                lines.append(f"- dedupe_count: {dedupe_count}")
            if coverage_fix_used is not None:
                lines.append(f"- coverage_fix_used: {'yes' if coverage_fix_used else 'no'}")
            if assumptions_added is not None:
                lines.append(f"- assumptions_added: {assumptions_added}")
            if constraints_added is not None:
                lines.append(f"- constraints_added: {constraints_added}")
            if assumptions_fixed is not None:
                lines.append(f"- assumptions_fixed: {'yes' if assumptions_fixed else 'no'}")
            if constraints_fixed is not None:
                lines.append(f"- constraints_fixed: {'yes' if constraints_fixed else 'no'}")
            if apply_format_retry_used is not None:
                lines.append(
                    f"- apply_format_retry_used: {'yes' if apply_format_retry_used else 'no'}"
                )
            if wrapper_repairs_applied is not None:
                lines.append(
                    f"- wrapper_repairs_applied: {'yes' if wrapper_repairs_applied else 'no'}"
                )
            if gemini_review_present is not None:
                lines.append(
                    f"- gemini_review_present: {'yes' if gemini_review_present else 'no'}"
                )
            if gemini_review_used is not None:
                lines.append(
                    f"- gemini_review_used: {'yes' if gemini_review_used else 'no'}"
                )
            if gemini_selected_model:
                lines.append(f"- gemini_selected_model: {gemini_selected_model}")
            if gemini_cross_review_skipped is not None:
                lines.append(
                    f"- gemini_cross_review_skipped: {'yes' if gemini_cross_review_skipped else 'no'}"
                )
            if gemini_error_summary:
                lines.append(f"- gemini_error_summary: {gemini_error_summary}")
            if gemini_cross_review_error:
                lines.append(f"- gemini_cross_review_error: {gemini_cross_review_error}")
            if cross_review_parse_error:
                lines.append(f"- cross_review_parse_error: {cross_review_parse_error}")
            if gemini_final_review_used is not None:
                lines.append(
                    f"- gemini_final_review_used: {'yes' if gemini_final_review_used else 'no'}"
                )
            if post_review_add_only_used is not None:
                lines.append(
                    f"- post_review_add_only_used: {'yes' if post_review_add_only_used else 'no'}"
                )
            if final_review_retry_used is not None:
                lines.append(
                    f"- final_review_retry_used: {'yes' if final_review_retry_used else 'no'}"
                )
            if apply_action_retry_used is not None:
                lines.append(
                    f"- apply_action_retry_used: {'yes' if apply_action_retry_used else 'no'}"
                )
            if coverage_unmapped_count is not None:
                lines.append(f"- coverage_unmapped_count: {coverage_unmapped_count}")
            if min_enforcement_unmet is not None:
                if min_enforcement_unmet:
                    lines.append("- minimum_enforcement: unmet (retries exhausted)")
                else:
                    lines.append("- minimum_enforcement: met")
            if isinstance(balance_results, dict) and balance_results:
                counts = balance_results.get("counts", {})
                missing = balance_results.get("missing", {})
                meets = balance_results.get("meets")
                lines.append(f"- balance_check_meets: {'yes' if meets else 'no'}")
                if counts:
                    lines.append(
                        "- balance_counts: "
                        + ", ".join(f"{key}={value}" for key, value in counts.items())
                    )
                if missing:
                    lines.append(
                        "- balance_missing: "
                        + ", ".join(f"{key}={value}" for key, value in missing.items())
                    )
            warnings_total = len(warnings) + len(self._requirements_warnings)
            lines.append(f"- warnings: {warnings_total}")
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
            adapter = GeminiAdapter()
            if hasattr(adapter, "get_diagnostics"):
                try:
                    diagnostics = adapter.get_diagnostics()  # type: ignore[attr-defined]
                    selected = diagnostics.get("selected_model")
                    if isinstance(selected, str):
                        self._gemini_selected_model = selected
                except Exception:
                    pass
            return adapter
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
        base_budget = self._artifact_token_budget("requirements", limits)
        apply_tokens = self._stage_max_tokens(limits, "requirements", "apply", base_budget)

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
            response = self._complete(adapter, full_prompt, apply_tokens)
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
                retry_response = self._complete(adapter, retry_full_prompt, apply_tokens)
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
            fix_response = self._complete(adapter, fix_full_prompt, apply_tokens)
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
        base_budget = self._artifact_token_budget("requirements", limits)
        apply_tokens = self._stage_max_tokens(limits, "requirements", "apply", base_budget)
        retry_response = self._complete(adapter, retry_full_prompt, apply_tokens)
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
            if path.name != "requirements_warnings.json":
                write_json(path.parent / "requirements_warnings.json", {"warnings": self._requirements_warnings})

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
        limits: RequirementsLimits,
    ) -> Dict:
        self._delta_retry_counts[retry_key] = self._delta_retry_counts.get(retry_key, 0) + 1
        prompt = read_text(self.prompts_dir / prompt_name)
        full_prompt = f"{prompt}\n\nINPUT:\n{json.dumps(payload)}\n"
        retry_prefix = f"turn_apply_delta_{retry_key}"
        write_text(raw_dir / f"{retry_prefix}_prompt.txt", full_prompt)
        base_budget = self._artifact_token_budget(retry_key, limits)
        apply_tokens = self._stage_max_tokens(limits, retry_key, "apply", base_budget)
        response = self._complete(adapter, full_prompt, apply_tokens)
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
        base_budget = self._artifact_token_budget(section_name, limits)
        lead_tokens = self._stage_max_tokens(limits, section_name, "lead", base_budget)
        retry_response = self._complete(adapter, retry_full_prompt, lead_tokens)
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
        acceptance_budget = self._artifact_configs()["acceptance_criteria"]["default_budget"]
        acceptance_tokens = self._apply_cli_cap(acceptance_budget)
        acceptance_response = self._complete(chatgpt, acceptance_full_prompt, acceptance_tokens)
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
        cross_response = self._complete(gemini, cross_full_prompt, acceptance_tokens)
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
        finalize_response = self._complete(chatgpt, finalize_full_prompt, acceptance_tokens)
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
        business_rules_tokens = self._stage_max_tokens(
            limits,
            "business_rules",
            "apply",
            self._artifact_token_budget("business_rules", limits),
        )
        workflows_tokens = self._stage_max_tokens(
            limits,
            "workflows",
            "apply",
            self._artifact_token_budget("workflows", limits),
        )
        combined_tokens = max(business_rules_tokens, workflows_tokens)
        prompt = self._render_prompt(
            read_text(self.prompts_dir / "requirements_apply_stage_b.md"), limits
        )
        payload = {"brief": brief, "requirements": final_requirements}
        full_prompt = f"{prompt}\n\nINPUT:\n{json.dumps(payload)}\n"
        write_text(raw_dir / "turn_apply_stage_b_prompt.txt", full_prompt)
        response = self._complete(adapter, full_prompt, combined_tokens)
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
                    limits=limits,
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
        domain_model_tokens = self._stage_max_tokens(
            limits,
            "domain_model",
            "apply",
            self._artifact_token_budget("domain_model", limits),
        )
        mvp_scope_tokens = self._stage_max_tokens(
            limits,
            "mvp_scope",
            "apply",
            self._artifact_token_budget("mvp_scope", limits),
        )
        combined_tokens = max(domain_model_tokens, mvp_scope_tokens)
        prompt = self._render_prompt(
            read_text(self.prompts_dir / "requirements_apply_stage_c.md"), limits
        )
        payload = {"brief": brief, "requirements": final_requirements}
        full_prompt = f"{prompt}\n\nINPUT:\n{json.dumps(payload)}\n"
        write_text(raw_dir / "turn_apply_stage_c_prompt.txt", full_prompt)
        response = self._complete(adapter, full_prompt, combined_tokens)
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
                    limits=limits,
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
        missing_areas: List[str] = []
        coverage_counts: Dict[str, int] = {}
        min_per_area = limits.min_per_area or 1
        if limits.coverage_areas:
            for area in limits.coverage_areas:
                keywords = self._coverage_keywords_for_area(limits, area)
                count = sum(
                    1
                    for item in req_items
                    if isinstance(item, dict)
                    and self._keyword_hits(str(item.get("text", "")), keywords) > 0
                )
                coverage_counts[area] = count
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
            if not any(
                seed_lower in str(item.get("text", "")).lower()
                for item in req_items
                if isinstance(item, dict)
            ):
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
            "coverage_counts": coverage_counts,
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
        base_budget = self._artifact_token_budget("requirements", limits)
        lead_tokens = self._stage_max_tokens(limits, "requirements", "lead", base_budget)
        retry_response = self._complete(adapter, retry_full_prompt, lead_tokens)
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
        token_budgets = frontmatter.get("token_budgets", {})
        if not isinstance(token_budgets, dict):
            token_budgets = {}
        lead_token_budgets: Dict[str, int] = {}
        apply_token_budgets: Dict[str, int] = {}
        for key, default_value in artifact_token_budgets.items():
            stage_budget = token_budgets.get(key, {})
            if not isinstance(stage_budget, dict):
                stage_budget = {}
            lead_value = stage_budget.get("lead_max_output_tokens")
            apply_value = stage_budget.get("apply_max_output_tokens")
            try:
                lead_token_budgets[key] = int(lead_value) if lead_value is not None else default_value
            except (TypeError, ValueError):
                lead_token_budgets[key] = default_value
            try:
                apply_token_budgets[key] = int(apply_value) if apply_value is not None else default_value
            except (TypeError, ValueError):
                apply_token_budgets[key] = default_value
        min_assumptions = targets.get(
            "min_assumptions",
            targets.get(
                "assumptions_min",
                frontmatter.get("min_assumptions", frontmatter.get("assumptions_min", 5)),
            ),
        )
        min_constraints = targets.get(
            "min_constraints",
            targets.get(
                "constraints_min",
                frontmatter.get("min_constraints", frontmatter.get("constraints_min", 5)),
            ),
        )
        try:
            assumptions_min = max(3, int(min_assumptions))
        except (TypeError, ValueError):
            assumptions_min = 5
        try:
            constraints_min = max(3, int(min_constraints))
        except (TypeError, ValueError):
            constraints_min = 5
        min_student_reqs_raw = targets.get("min_student_reqs", 25)
        min_coordinator_reqs_raw = targets.get("min_coordinator_reqs", 15)
        min_admin_reqs_raw = targets.get("min_admin_reqs", 8)
        min_domain_keyword_hits_raw = targets.get("min_domain_keyword_hits", 40)
        final_target_raw = frontmatter.get(
            "final_target_items",
            targets.get("final_target_items", targets.get("target_final_items")),
        )
        add_only_batch_raw = frontmatter.get("add_only_batch_size", 15)
        add_only_rounds_raw = frontmatter.get("add_only_max_rounds", 6)
        add_only_min_new_per_area_raw = frontmatter.get(
            "add_only_min_new_per_area",
            targets.get("add_only_min_new_per_area"),
        )
        try:
            min_student_reqs = int(min_student_reqs_raw)
        except (TypeError, ValueError):
            min_student_reqs = 25
        try:
            min_coordinator_reqs = int(min_coordinator_reqs_raw)
        except (TypeError, ValueError):
            min_coordinator_reqs = 15
        try:
            min_admin_reqs = int(min_admin_reqs_raw)
        except (TypeError, ValueError):
            min_admin_reqs = 8
        try:
            min_domain_keyword_hits = int(min_domain_keyword_hits_raw)
        except (TypeError, ValueError):
            min_domain_keyword_hits = 40
        try:
            final_target_items = int(final_target_raw) if final_target_raw is not None else None
        except (TypeError, ValueError):
            final_target_items = None
        try:
            add_only_batch_size = max(1, int(add_only_batch_raw))
        except (TypeError, ValueError):
            add_only_batch_size = 15
        try:
            add_only_max_rounds = int(add_only_rounds_raw)
        except (TypeError, ValueError):
            add_only_max_rounds = 6
        try:
            add_only_min_new_per_area = (
                int(add_only_min_new_per_area_raw)
                if add_only_min_new_per_area_raw is not None
                else None
            )
        except (TypeError, ValueError):
            add_only_min_new_per_area = None

        coverage_keywords: Dict[str, List[str]] = {
            area: list(keywords)
            for area, keywords in self._DEFAULT_COVERAGE_KEYWORDS.items()
        }
        coverage_prefix_mode_raw = frontmatter.get("coverage_prefix_mode", False)
        if isinstance(coverage_prefix_mode_raw, str):
            coverage_prefix_mode = coverage_prefix_mode_raw.strip().lower() in {"1", "true", "yes"}
        else:
            coverage_prefix_mode = bool(coverage_prefix_mode_raw)
        coverage_areas_raw = frontmatter.get("coverage_areas")
        coverage_areas = []
        if isinstance(coverage_areas_raw, str):
            coverage_areas_raw = [coverage_areas_raw]
        if isinstance(coverage_areas_raw, list):
            for entry in coverage_areas_raw:
                if isinstance(entry, str):
                    coverage_areas.append(entry)
                    if entry not in coverage_keywords:
                        coverage_keywords[entry] = []
                elif isinstance(entry, dict):
                    name = entry.get("name")
                    keywords = entry.get("keywords", [])
                    if isinstance(name, str):
                        coverage_areas.append(name)
                        if not isinstance(keywords, list):
                            keywords = []
                        if keywords:
                            coverage_keywords[name] = [str(item) for item in keywords if str(item)]
                        elif name not in coverage_keywords:
                            coverage_keywords[name] = []
        if not coverage_areas:
            coverage_areas = list(self._DEFAULT_COVERAGE_KEYWORDS.keys())
        normalized_areas: List[str] = []
        for entry in coverage_areas:
            if isinstance(entry, str):
                normalized_areas.append(entry)
        coverage_areas = normalized_areas

        return RequirementsLimits(
            req_min=req_min,
            req_max=req_max_int,
            final_target_items=final_target_items,
            add_only_batch_size=add_only_batch_size,
            add_only_max_rounds=add_only_max_rounds,
            add_only_min_new_per_area=add_only_min_new_per_area,
            assumptions_min=assumptions_min,
            constraints_min=constraints_min,
            min_student_reqs=min_student_reqs,
            min_coordinator_reqs=min_coordinator_reqs,
            min_admin_reqs=min_admin_reqs,
            min_domain_keyword_hits=min_domain_keyword_hits,
            roles_expected=list(frontmatter.get("roles_expected", [])),
            coverage_areas=coverage_areas,
            coverage_keywords=coverage_keywords,
            min_per_area=min_per_area_int,
            coverage_prefix_mode=coverage_prefix_mode,
            seed_requirements=seed_requirements,
            requested_artifacts=requested_list,
            artifact_token_budgets=artifact_token_budgets,
            lead_token_budgets=lead_token_budgets,
            apply_token_budgets=apply_token_budgets,
        )

    def _limits_payload(self, limits: RequirementsLimits) -> Dict:
        return {
            "requirements_min": limits.req_min,
            "requirements_max": limits.req_max,
            "assumptions_min": limits.assumptions_min,
            "constraints_min": limits.constraints_min,
            "min_student_reqs": limits.min_student_reqs,
            "min_coordinator_reqs": limits.min_coordinator_reqs,
            "min_admin_reqs": limits.min_admin_reqs,
            "min_domain_keyword_hits": limits.min_domain_keyword_hits,
            "domain_keywords": self._DOMAIN_KEYWORDS,
            "roles_expected": limits.roles_expected,
            "coverage_areas": limits.coverage_areas,
            "min_per_area": limits.min_per_area,
            "coverage_prefix_mode": limits.coverage_prefix_mode,
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
