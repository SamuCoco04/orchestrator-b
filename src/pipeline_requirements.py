from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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
class GateResult:
    failures: List[str]


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
        lead_response_path = raw_dir / "turnr1_lead_raw.txt"
        write_text(lead_response_path, lead_response.raw_text)
        self._write_usage(raw_dir / "turnr1_chatgpt_lead_usage.json", lead_response)

        wrapper = self._try_parse_wrapper(lead_response.raw_text)
        wrapper_keys: List[str] | None = None
        if wrapper is not None:
            wrapper_keys = sorted(wrapper.keys())
            review_json = wrapper.get("REVIEW_JSON")
            draft_requirements = wrapper.get("REQUIREMENTS_JSON")
        else:
            review_json = self._extract_marked_json(
                lead_response.raw_text,
                "REVIEW_JSON:",
                {"accepted", "rejected", "issues", "missing", "rationale"},
            )
            draft_requirements = None

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
        except ValueError:
            retry_prompt = read_text(
                self.prompts_dir / "requirements_lead_retry_requirements_only.md"
            )
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
            except ValueError as exc:
                wrapper_note = ""
                if wrapper_keys is not None:
                    wrapper_note = f" Detected wrapper JSON with keys: {wrapper_keys}."
                raise RuntimeError(
                    "Lead returned only REVIEW_JSON; enforce prompt format."
                    + wrapper_note
                ) from exc

        requirements_schema = self._load_schema("requirements_strict.schema.json")
        validate(instance=draft_requirements, schema=requirements_schema)
        review_schema = self._load_schema("requirements_review.schema.json")
        validate(instance=review_json, schema=review_schema)

        write_json(raw_dir / "turnr1_draft.json", draft_requirements)
        write_json(raw_dir / "turnr1_review.json", review_json)
        write_json(artifacts_dir / "requirements_review.json", review_json)

        cross_prompt = read_text(self.prompts_dir / "requirements_gemini_cross_review.md")
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

        apply_prompt = read_text(self.prompts_dir / "requirements_apply_chatgpt.md")
        final_requirements, changelog = self._run_apply(
            chatgpt,
            apply_prompt,
            brief,
            draft_requirements,
            review_json,
            cross_review,
            raw_dir,
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
            "metrics": {
                "requirements_count": len(final_requirements.get("requirements", [])),
                "assumptions_count": len(final_requirements.get("assumptions", [])),
                "constraints_count": len(final_requirements.get("constraints", [])),
                "target_min": int(self._env("ORCH_REQ_MIN_COUNT", "30")),
                "target_max": int(self._env("ORCH_REQ_MAX_COUNT", "45")),
            },
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

        return {
            "draft": draft_requirements,
            "review": review_json,
            "cross_review": cross_review,
            "final": final_requirements,
            "changelog": changelog,
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
        raw_dir: Path,
    ) -> tuple[Dict, Dict]:
        failures: List[str] = []
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
                "gates": self._gate_config(),
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
            try:
                changelog = self._extract_marked_json(
                    response.raw_text,
                    "CHANGELOG_JSON:",
                    {"splits", "replacements", "added", "removed"},
                )
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
                except ValueError as exc:
                    raise RuntimeError("Missing CHANGELOG_JSON in apply output.") from exc

            requirements_schema = self._load_schema("requirements_strict.schema.json")
            validate(instance=final_requirements, schema=requirements_schema)
            self._validate_changelog(changelog)

            failures = self._run_gates(final_requirements, review, changelog)
            if not failures:
                return final_requirements, changelog

            fix_prompt = read_text(
                self.prompts_dir / "requirements_apply_retry_fix_gates.md"
            )
            fix_payload = {
                "brief": brief,
                "previous": {
                    "final_requirements": final_requirements,
                    "changelog": changelog,
                },
                "review": review,
                "cross_review": cross_review,
                "failures": failures,
            }
            fix_full_prompt = f"{fix_prompt}\n\nINPUT:\n{json.dumps(fix_payload)}\n"
            fix_prompt_path = raw_dir / "turnr4_apply_retry_fix_prompt.txt"
            write_text(fix_prompt_path, fix_full_prompt)
            fix_response = adapter.complete(fix_full_prompt)
            write_text(raw_dir / "turnr4_apply_retry_fix_raw.txt", fix_response.raw_text)
            self._write_usage(raw_dir / "turnr4_apply_retry_fix_usage.json", fix_response)

            final_requirements = self._extract_marked_json(
                fix_response.raw_text,
                "FINAL_REQUIREMENTS_JSON:",
                {"requirements", "assumptions", "constraints"},
            )
            changelog = self._extract_marked_json(
                fix_response.raw_text,
                "CHANGELOG_JSON:",
                {"splits", "replacements", "added", "removed"},
            )
            requirements_schema = self._load_schema("requirements_strict.schema.json")
            validate(instance=final_requirements, schema=requirements_schema)
            self._validate_changelog(changelog)

            failures = self._run_gates(final_requirements, review, changelog)
            if not failures:
                return final_requirements, changelog

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
                return candidate

        match = re.search(re.escape(marker), raw_text)
        if match:
            snippet = raw_text[match.end():]
            parsed = self._match_expected_json(snippet, expected_keys)
            if parsed is not None:
                return parsed

        parsed = self._match_expected_json(raw_text, expected_keys)
        if parsed is not None:
            return parsed

        for obj in self._scan_json_objects(raw_text):
            if self._matches_keys(obj, expected_keys):
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
            "REVIEW_JSON" in parsed or "REQUIREMENTS_JSON" in parsed
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

    def _run_gates(self, requirements: Dict, review: Dict, changelog: Dict) -> List[str]:
        failures: List[str] = []
        min_count = int(self._env("ORCH_REQ_MIN_COUNT", "30"))
        max_count = int(self._env("ORCH_REQ_MAX_COUNT", "45"))
        enforce_splits = self._env("ORCH_ENFORCE_SPLITS", "false").lower() == "true"

        req_items = requirements.get("requirements", [])
        req_count = len(req_items)
        if req_count < min_count or req_count > max_count:
            failures.append(
                f"Gate A: requirements count {req_count} not in {min_count}..{max_count}"
            )

        if len(requirements.get("assumptions", [])) < 3:
            failures.append("Gate B: assumptions must have at least 3 items")
        if len(requirements.get("constraints", [])) < 3:
            failures.append("Gate B: constraints must have at least 3 items")

        rejected_ids = set(review.get("rejected", []))
        accepted_ids = set(review.get("accepted", []))
        added_ids = set(changelog.get("added", []))
        split_targets = {
            item_id
            for split in changelog.get("splits", [])
            for item_id in split.get("to", [])
        }
        added_ids |= split_targets
        final_ids = {item.get("id") for item in req_items}

        if rejected_ids.intersection(final_ids):
            failures.append("Gate C: rejected requirement IDs present in final")

        missing_from_acceptance = {
            req_id
            for req_id in final_ids
            if req_id not in accepted_ids and req_id not in added_ids
        }
        if missing_from_acceptance:
            failures.append("Gate C: final requirements include IDs not accepted or added")

        forbidden_ids = {"REQ-1", "REQ-7", "REQ-12", "REQ-14"}
        if enforce_splits:
            invalid = sorted(forbidden_ids.intersection(final_ids))
            if invalid:
                failures.append(f"Gate D: forbidden IDs present {invalid}")

        issues = " ".join(review.get("issues", [])).lower()
        needs_split = any(word in issues for word in ["split", "epic"])
        if needs_split:
            splits = changelog.get("splits", []) if isinstance(changelog, dict) else []
            if not splits:
                failures.append("Gate E: review mentions split/epic but changelog has no splits")

        return failures

    def _validate_changelog(self, changelog: Dict) -> None:
        required_keys = {"splits", "replacements", "added", "removed"}
        if not isinstance(changelog, dict) or not required_keys.issubset(changelog.keys()):
            raise ValueError("Changelog JSON missing required keys.")

    def _gate_config(self) -> Dict:
        return {
            "min_count": int(self._env("ORCH_REQ_MIN_COUNT", "30")),
            "enforce_splits": self._env("ORCH_ENFORCE_SPLITS", "false"),
        }

    def _env(self, key: str, default: str) -> str:
        return os.getenv(key, default)

    def _write_usage(self, path: Path, response: LLMResponse) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            write_json(path, usage)

    def _load_schema(self, name: str) -> Dict:
        return json.loads(read_text(self.schemas_dir / name))
