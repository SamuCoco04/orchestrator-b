from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from jsonschema import validate

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
    min_per_area: int | None
    seed_requirements: List[str]
    requested_artifacts: List[str]


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

    def run(self, brief_path: Path, run_dir: Path) -> Dict[str, Dict]:
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
        raw_brief = read_text(brief_path)
        frontmatter, brief = self._parse_frontmatter(raw_brief)
        limits = self._limits_from_frontmatter(frontmatter)

        raw_dir = run_dir / "raw"
        artifacts_dir = run_dir / "artifacts"
        adrs_dir = artifacts_dir / "adrs"
        raw_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        adrs_dir.mkdir(parents=True, exist_ok=True)

        gemini = self._adapter("gemini")
        chatgpt = self._adapter("chatgpt")

        lead_template = read_text(self.prompts_dir / "requirements_chatgpt_lead.md")
        lead_prompt = self._render_prompt(lead_template, limits)
        lead_full_prompt = f"{lead_prompt}\n\nINPUT:\n{brief}\n"
        lead_prompt_path = raw_dir / "turnr1_chatgpt_lead_prompt.txt"
        write_text(lead_prompt_path, lead_full_prompt)
        lead_response = chatgpt.complete(lead_full_prompt)
        lead_response_path = raw_dir / "turnr1_lead_raw.txt"
        write_text(lead_response_path, lead_response.raw_text)
        self._write_usage(raw_dir / "turnr1_chatgpt_lead_usage.json", lead_response)

        wrapper = self._try_parse_wrapper(lead_response.raw_text)
        wrapper_keys: List[str] | None = None
        if wrapper is not None:
            wrapper_keys = sorted(wrapper.keys())
            review_json = wrapper.get("REVIEW_JSON")
            draft_requirements = wrapper.get("REQUIREMENTS_JSON")
            business_rules = wrapper.get("BUSINESS_RULES_JSON")
            workflows = wrapper.get("WORKFLOWS_JSON")
            domain_model = wrapper.get("DOMAIN_MODEL_JSON")
            mvp_scope = wrapper.get("MVP_SCOPE_JSON")
        else:
            review_json = self._extract_marked_json(
                lead_response.raw_text,
                "REVIEW_JSON:",
                {"accepted", "rejected", "issues", "missing", "rationale"},
            )
            draft_requirements = None
            business_rules = None
            workflows = None
            domain_model = None
            mvp_scope = None

        try:
            if draft_requirements is None:
                draft_requirements = self._extract_marked_json(
                    lead_response.raw_text,
                    "REQUIREMENTS_JSON:",
                    {"requirements", "assumptions", "constraints"},
                )
            if review_json is None:
                review_json = self._extract_marked_json(
                    lead_response.raw_text,
                    "REVIEW_JSON:",
                    {"accepted", "rejected", "issues", "missing", "rationale"},
                )
            if self._is_requested("business_rules", limits) and business_rules is None:
                business_rules = self._extract_marked_json(
                    lead_response.raw_text,
                    "BUSINESS_RULES_JSON:",
                    {"rules"},
                )
            if self._is_requested("workflows", limits) and workflows is None:
                workflows = self._extract_marked_json(
                    lead_response.raw_text,
                    "WORKFLOWS_JSON:",
                    {"workflows"},
                )
            if self._is_requested("domain_model", limits) and domain_model is None:
                domain_model = self._extract_marked_json(
                    lead_response.raw_text,
                    "DOMAIN_MODEL_JSON:",
                    {"entities", "relationships"},
                )
            if self._is_requested("mvp_scope", limits) and mvp_scope is None:
                mvp_scope = self._extract_marked_json(
                    lead_response.raw_text,
                    "MVP_SCOPE_JSON:",
                    {"in_scope", "out_of_scope"},
                )
        except ValueError:
            retry_template = read_text(
                self.prompts_dir / "requirements_lead_retry_requirements_only.md"
            )
            retry_prompt = self._render_prompt(retry_template, limits)
            retry_full_prompt = f"{retry_prompt}\n\nINPUT:\n{brief}\n"
            retry_prompt_path = raw_dir / "turnr1_lead_retry_prompt.txt"
            write_text(retry_prompt_path, retry_full_prompt)
            retry_response = chatgpt.complete(retry_full_prompt)
            retry_raw_path = raw_dir / "turnr1_lead_retry_raw.txt"
            write_text(retry_raw_path, retry_response.raw_text)
            self._write_usage(raw_dir / "turnr1_lead_retry_usage.json", retry_response)
            try:
                draft_requirements = self._extract_marked_json(
                    retry_response.raw_text,
                    "REQUIREMENTS_JSON:",
                    {"requirements", "assumptions", "constraints"},
                )
                write_json(
                    raw_dir / "turnr1_lead_retry_requirements_extracted.json",
                    draft_requirements,
                )
            except ValueError as exc:
                wrapper_note = ""
                if wrapper_keys is not None:
                    wrapper_note = f" Detected wrapper JSON with keys: {wrapper_keys}."
                raise RuntimeError(
                    "Lead returned only REVIEW_JSON; enforce prompt format."
                    + wrapper_note
                ) from exc

        if self._is_requested("business_rules", limits) and business_rules is None:
            business_rules = self._require_section(
                section_name="business_rules",
                label="BUSINESS_RULES_JSON",
                schema_name="business_rules.schema.json",
                retry_prompt_path="requirements_lead_retry_business_rules.md",
                default_value={"rules": []},
                expected_keys={"rules"},
                brief=brief,
                limits=limits,
                adapter=chatgpt,
                raw_dir=raw_dir,
                artifacts_dir=artifacts_dir,
            )
        if self._is_requested("workflows", limits) and workflows is None:
            workflows = self._require_section(
                section_name="workflows",
                label="WORKFLOWS_JSON",
                schema_name="workflows.schema.json",
                retry_prompt_path="requirements_lead_retry_workflows.md",
                default_value={"workflows": []},
                expected_keys={"workflows"},
                brief=brief,
                limits=limits,
                adapter=chatgpt,
                raw_dir=raw_dir,
                artifacts_dir=artifacts_dir,
            )
        if self._is_requested("domain_model", limits) and domain_model is None:
            domain_model = self._require_section(
                section_name="domain_model",
                label="DOMAIN_MODEL_JSON",
                schema_name="domain_model.schema.json",
                retry_prompt_path="requirements_lead_retry_domain_model.md",
                default_value={"entities": [], "relationships": []},
                expected_keys={"entities", "relationships"},
                brief=brief,
                limits=limits,
                adapter=chatgpt,
                raw_dir=raw_dir,
                artifacts_dir=artifacts_dir,
            )
        if self._is_requested("mvp_scope", limits) and mvp_scope is None:
            mvp_scope = self._require_section(
                section_name="mvp_scope",
                label="MVP_SCOPE_JSON",
                schema_name="mvp_scope.schema.json",
                retry_prompt_path="requirements_lead_retry_mvp_scope.md",
                default_value={"in_scope": [], "out_of_scope": [], "milestones": []},
                expected_keys={"in_scope", "out_of_scope"},
                brief=brief,
                limits=limits,
                adapter=chatgpt,
                raw_dir=raw_dir,
                artifacts_dir=artifacts_dir,
            )

        review_extracted = review_json
        draft_extracted = draft_requirements
        write_json(raw_dir / "turnr1_review_extracted.json", review_extracted)
        write_json(raw_dir / "turnr1_requirements_extracted.json", draft_extracted)
        if business_rules is not None:
            write_json(raw_dir / "turnr1_business_rules_extracted.json", business_rules)
        if workflows is not None:
            write_json(raw_dir / "turnr1_workflows_extracted.json", workflows)
        if domain_model is not None:
            write_json(raw_dir / "turnr1_domain_model_extracted.json", domain_model)
        if mvp_scope is not None:
            write_json(raw_dir / "turnr1_mvp_scope_extracted.json", mvp_scope)

        review_json = self.normalize_review_json(review_json)
        if self._review_normalization_warnings:
            write_json(
                artifacts_dir / "review_normalization_warnings.json",
                {"warnings": self._review_normalization_warnings},
            )

        draft_requirements, string_count = self._normalize_requirements_payload(
            draft_requirements, stage="draft"
        )
        if string_count > 2:
            retry_requirements = self._retry_requirements_only(
                chatgpt,
                brief,
                limits,
                raw_dir,
                "requirements_lead_retry_requirements_only.md",
                "turnr1_requirements_retry",
                "REQUIREMENTS_JSON:",
                {"requirements", "assumptions", "constraints"},
            )
            if retry_requirements is not None:
                draft_requirements, _ = self._normalize_requirements_payload(
                    retry_requirements, stage="draft_retry"
                )

        write_json(raw_dir / "turnr1_requirements_normalized.json", draft_requirements)
        self._write_requirements_warnings(artifacts_dir / "warnings.json")

        requirements_schema = self._load_schema("normalized_requirements.schema.json")
        validate(instance=draft_requirements, schema=requirements_schema)
        review_schema = self._load_schema("requirements_review.schema.json")
        try:
            validate(instance=review_json, schema=review_schema)
        except Exception as exc:
            write_json(raw_dir / "turnr1_review_invalid_raw.json", review_extracted)
            write_json(
                artifacts_dir / "review_invalid_normalized.json",
                review_json,
            )
            raise exc

        draft_requirements, id_map = self._normalize_requirements(draft_requirements)
        review_json = self._normalize_review(review_json, id_map)

        write_json(raw_dir / "turnr1_draft.json", draft_requirements)
        write_json(raw_dir / "turnr1_review.json", review_json)
        write_json(artifacts_dir / "requirements_review.json", review_json)

        if business_rules is not None:
            business_rules = self._normalize_business_rules(business_rules)
            self._validate_artifact(
                business_rules, "business_rules.schema.json", "BUSINESS_RULES_JSON"
            )
            write_json(artifacts_dir / "business_rules.json", business_rules)
            self._write_business_rules_markdown(
                artifacts_dir / "business_rules.md", business_rules
            )
        if workflows is not None:
            workflows = self._normalize_workflows(workflows)
            write_json(artifacts_dir / "workflows_normalized.json", workflows)
            self._validate_artifact(
                workflows, "workflows.schema.json", "WORKFLOWS_JSON"
            )
            write_json(artifacts_dir / "workflows.json", workflows)
            self._write_workflows_markdown(artifacts_dir / "workflows.md", workflows)
        if domain_model is not None:
            self._validate_artifact(
                domain_model, "domain_model.schema.json", "DOMAIN_MODEL_JSON"
            )
            write_json(artifacts_dir / "domain_model.json", domain_model)
            self._write_domain_model_markdown(
                artifacts_dir / "domain_model.md", domain_model
            )
        if mvp_scope is not None:
            self._validate_artifact(
                mvp_scope, "mvp_scope.schema.json", "MVP_SCOPE_JSON"
            )
            write_json(artifacts_dir / "mvp_scope.json", mvp_scope)
            self._write_mvp_scope_markdown(
                artifacts_dir / "mvp_scope.md", mvp_scope
            )

        coverage = self._check_coverage(draft_requirements, limits)
        repairs_applied = False
        if coverage["needs_retry"]:
            retries = self._run_coverage_retry(
                chatgpt, brief, draft_requirements, coverage, limits, raw_dir
            )
            if retries is not None:
                repairs_applied = True
                draft_requirements = self._merge_requirements(draft_requirements, retries)
                draft_requirements, id_map = self._normalize_requirements(draft_requirements)
                review_json = self._normalize_review(review_json, id_map)
                write_json(
                    raw_dir / "turnr1_coverage_requirements_normalized.json",
                    draft_requirements,
                )
                coverage = self._check_coverage(draft_requirements, limits)

        cross_template = read_text(self.prompts_dir / "requirements_gemini_cross_review.md")
        cross_prompt = self._render_prompt(cross_template, limits)
        cross_payload = {
            "brief": brief,
            "requirements": draft_requirements,
            "review": review_json,
        }
        cross_full_prompt = f"{cross_prompt}\n\nINPUT:\n{json.dumps(cross_payload)}\n"
        cross_prompt_path = raw_dir / "turnr3_gemini_cross_review_prompt.txt"
        write_text(cross_prompt_path, cross_full_prompt)
        cross_response = gemini.complete(cross_full_prompt)
        cross_response_path = raw_dir / "turnr3_gemini_cross_review_response.txt"
        write_text(cross_response_path, cross_response.raw_text)
        write_text(raw_dir / "turnr3_gemini_cross_review_raw.txt", cross_response.raw_text)
        self._write_usage(raw_dir / "turnr3_gemini_cross_review_usage.json", cross_response)

        cross_review = extract_json(cross_response.raw_text)
        if not isinstance(cross_review, dict):
            raise ValueError("Turn R3 cross-review output must be a JSON object.")
        write_json(raw_dir / "turnr3_gemini_cross_review.json", cross_review)
        write_json(artifacts_dir / "turnr3_gemini_cross_review.json", cross_review)

        apply_template = read_text(self.prompts_dir / "requirements_apply_stage_a.md")
        apply_prompt = self._render_prompt(apply_template, limits)
        final_requirements, changelog, apply_repairs = self._run_apply(
            chatgpt,
            apply_prompt,
            brief,
            draft_requirements,
            review_json,
            cross_review,
            limits,
            raw_dir,
            artifacts_dir,
        )
        repairs_applied = repairs_applied or apply_repairs
        final_coverage = self._check_coverage(final_requirements, limits)

        final_artifacts = self._run_apply_stage_b(
            chatgpt,
            brief,
            final_requirements,
            limits,
            raw_dir,
            artifacts_dir,
        )
        final_artifacts.update(
            self._run_apply_stage_c(
                chatgpt,
                brief,
                final_requirements,
                limits,
                raw_dir,
                artifacts_dir,
            )
        )

        write_json(raw_dir / "turnr4_final_requirements.json", final_requirements)
        write_json(raw_dir / "turnr4_changelog.json", changelog)
        write_json(artifacts_dir / "requirements.json", final_requirements)
        write_json(artifacts_dir / "changelog.json", changelog)

        write_requirements(artifacts_dir / "requirements.md", final_requirements)

        adr_prompt = read_text(self.prompts_dir / "requirements_chatgpt_adr.md")
        adr_payload = {
            "requirements": final_requirements,
            "review": review_json,
            "cross_review": cross_review,
            "changelog": changelog,
            "metrics": self._metrics(final_requirements, limits),
        }
        adr_full_prompt = f"{adr_prompt}\n\nINPUT:\n{json.dumps(adr_payload)}\n"
        adr_prompt_path = raw_dir / "turnr5_chatgpt_adr_prompt.txt"
        write_text(adr_prompt_path, adr_full_prompt)
        adr_response = chatgpt.complete(adr_full_prompt)
        adr_response_path = raw_dir / "turnr5_chatgpt_adr_response.txt"
        write_text(adr_response_path, adr_response.raw_text)
        self._write_usage(raw_dir / "turnr5_chatgpt_adr_usage.json", adr_response)

        adr = extract_json(adr_response.raw_text)
        adr_schema = self._load_schema("adr.schema.json")
        validate(instance=adr, schema=adr_schema)
        write_json(artifacts_dir / "turnr5_chatgpt_adr.json", adr)
        write_adr(adrs_dir / f"{adr['adr_id']}.md", adr)

        acceptance_criteria = None
        if self._is_requested("acceptance_criteria", limits):
            acceptance_criteria = self._run_acceptance_criteria(
                chatgpt,
                gemini,
                final_requirements,
                raw_dir,
                artifacts_dir,
            )

        self._write_run_summary(
            artifacts_dir,
            final_requirements,
            final_artifacts,
            acceptance_criteria,
            final_coverage,
            repairs_applied,
            [
                lead_response,
                cross_response,
                adr_response,
            ],
        )

        return {
            "draft": draft_requirements,
            "review": review_json,
            "cross_review": cross_review,
            "final": final_requirements,
            "changelog": changelog,
            "final_artifacts": final_artifacts,
            "acceptance_criteria": acceptance_criteria,
            "adr": adr,
        }

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

    def _validate_artifact(self, payload: Dict, schema_name: str, label: str) -> None:
        schema = self._load_schema(schema_name)
        try:
            validate(instance=payload, schema=schema)
        except Exception as exc:
            snippet = json.dumps(payload)[:300]
            raise RuntimeError(
                f"{label} failed validation. Snippet: {snippet}"
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
            workflows = self._normalize_workflows(workflows)
            write_json(artifacts_dir / "workflows_normalized.json", workflows)
            self._validate_artifact(
                workflows, "workflows.schema.json", "FINAL_WORKFLOWS_JSON"
            )
            results["workflows"] = workflows
            write_json(artifacts_dir / "workflows.json", workflows)
            self._write_workflows_markdown(artifacts_dir / "workflows.md", workflows)
        except Exception as exc:
            self._section_warnings.setdefault("workflows", []).append(str(exc))
            write_json(
                artifacts_dir / "workflows_warnings.json",
                {"warnings": self._section_warnings["workflows"]},
            )
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
            self._validate_artifact(
                domain_model, "domain_model.schema.json", "FINAL_DOMAIN_MODEL_JSON"
            )
            results["domain_model"] = domain_model
            write_json(artifacts_dir / "domain_model.json", domain_model)
            self._write_domain_model_markdown(
                artifacts_dir / "domain_model.md", domain_model
            )
        except Exception as exc:
            self._section_warnings.setdefault("domain_model", []).append(str(exc))
            write_json(
                artifacts_dir / "domain_model_warnings.json",
                {"warnings": self._section_warnings["domain_model"]},
            )
            results["domain_model"] = {"entities": [], "relationships": []}

        try:
            mvp_scope = self._extract_marked_json(
                response.raw_text,
                "FINAL_MVP_SCOPE_JSON:",
                {"in_scope", "out_of_scope"},
            )
            write_json(raw_dir / "turn_apply_stage_c_mvp_scope_extracted.json", mvp_scope)
            self._validate_artifact(
                mvp_scope, "mvp_scope.schema.json", "FINAL_MVP_SCOPE_JSON"
            )
            results["mvp_scope"] = mvp_scope
            write_json(artifacts_dir / "mvp_scope.json", mvp_scope)
            self._write_mvp_scope_markdown(artifacts_dir / "mvp_scope.md", mvp_scope)
        except Exception as exc:
            self._section_warnings.setdefault("mvp_scope", []).append(str(exc))
            write_json(
                artifacts_dir / "mvp_scope_warnings.json",
                {"warnings": self._section_warnings["mvp_scope"]},
            )
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
                candidate = f"REQ-TEMP-{next_id}"
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
        req_min = int(frontmatter.get("target_min_reqs", req_target.get("min", 30)))
        req_max = frontmatter.get("target_max_reqs", req_target.get("max"))
        if req_max == "" or req_max is None:
            req_max_int = None
        else:
            req_max_int = int(req_max)
        coverage_areas = frontmatter.get("coverage_areas", [])
        if isinstance(coverage_areas, str):
            coverage_areas = [coverage_areas]
        coverage_areas = list(coverage_areas)
        min_per_area = frontmatter.get("min_per_area")
        min_per_area_int = int(min_per_area) if min_per_area is not None else None
        seed_requirements = frontmatter.get("seed_requirements", [])
        if isinstance(seed_requirements, str):
            seed_requirements = [seed_requirements]
        seed_requirements = list(seed_requirements)
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
        return RequirementsLimits(
            req_min=req_min,
            req_max=req_max_int,
            assumptions_min=int(frontmatter.get("assumptions_min", 3)),
            constraints_min=int(frontmatter.get("constraints_min", 3)),
            roles_expected=list(frontmatter.get("roles_expected", [])),
            coverage_areas=coverage_areas,
            min_per_area=min_per_area_int,
            seed_requirements=seed_requirements,
            requested_artifacts=requested_list,
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
