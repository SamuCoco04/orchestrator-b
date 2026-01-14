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


class RequirementsPipeline:
    def __init__(self, mode: str, base_dir: Path) -> None:
        self.mode = mode
        self.base_dir = base_dir
        self.schemas_dir = base_dir / "schemas"
        self.prompts_dir = base_dir / "configs" / "prompts"
        self._review_normalization_warnings: List[str] = []

    def run(self, brief_path: Path, run_dir: Path) -> Dict[str, Dict]:
        self._review_normalization_warnings = []
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
            except ValueError as exc:
                wrapper_note = ""
                if wrapper_keys is not None:
                    wrapper_note = f" Detected wrapper JSON with keys: {wrapper_keys}."
                raise RuntimeError(
                    "Lead returned only REVIEW_JSON; enforce prompt format."
                    + wrapper_note
                ) from exc

        review_json = self.normalize_review_json(review_json)
        if self._review_normalization_warnings:
            write_json(
                artifacts_dir / "review_normalization_warnings.json",
                {"warnings": self._review_normalization_warnings},
            )

        requirements_schema = self._load_schema("normalized_requirements.schema.json")
        validate(instance=draft_requirements, schema=requirements_schema)
        review_schema = self._load_schema("requirements_review.schema.json")
        validate(instance=review_json, schema=review_schema)

        draft_requirements, id_map = self._normalize_requirements(draft_requirements)
        review_json = self._normalize_review(review_json, id_map)

        write_json(raw_dir / "turnr1_draft.json", draft_requirements)
        write_json(raw_dir / "turnr1_review.json", review_json)
        write_json(artifacts_dir / "requirements_review.json", review_json)

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

        apply_template = read_text(self.prompts_dir / "requirements_apply_chatgpt.md")
        apply_prompt = self._render_prompt(apply_template, limits)
        final_requirements, changelog = self._run_apply(
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
        limits: RequirementsLimits,
        raw_dir: Path,
        artifacts_dir: Path,
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
            changelog_raw_context: Dict | None = None
            try:
                changelog = self._extract_marked_json(
                    response.raw_text,
                    "CHANGELOG_JSON:",
                    {"splits", "replacements", "added", "removed"},
                )
                changelog_raw_context = changelog
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
                except ValueError as exc:
                    changelog = None
                    changelog_raw_context = {"raw_response": retry_response.raw_text}

            requirements_schema = self._load_schema("normalized_requirements.schema.json")
            validate(instance=final_requirements, schema=requirements_schema)

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
                }
            changelog = self._recompute_added(changelog, final_requirements, review)

            failures = self._run_gates(final_requirements, review, changelog, limits)
            if not failures:
                return final_requirements, changelog

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
            try:
                repair_changelog = self._extract_marked_json(
                    fix_response.raw_text,
                    "CHANGELOG_JSON:",
                    {"splits", "replacements", "added", "removed"},
                )
            except ValueError:
                repair_changelog = {
                    "warnings": ["Missing CHANGELOG_JSON in apply fix output."],
                }
            validate(instance=repair_requirements, schema=requirements_schema)

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
                }
            changelog = self._recompute_added(changelog, final_requirements, review)

            failures = self._run_gates(final_requirements, review, changelog, limits)
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
            mapped = [map_id(item) for item in values if isinstance(item, str)]
            return [item for item in mapped if item]

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

    def _validate_changelog(self, changelog: Dict) -> None:
        required_keys = {"splits", "replacements", "added", "removed"}
        if not isinstance(changelog, dict) or not required_keys.issubset(changelog.keys()):
            raise ValueError("Changelog JSON missing required keys.")

    def _limits_from_frontmatter(self, frontmatter: Dict) -> RequirementsLimits:
        req_target = frontmatter.get("requirements_target", {}) if isinstance(frontmatter, dict) else {}
        req_min = int(req_target.get("min", 30))
        req_max = req_target.get("max")
        req_max_int = int(req_max) if req_max is not None else None
        return RequirementsLimits(
            req_min=req_min,
            req_max=req_max_int,
            assumptions_min=int(frontmatter.get("assumptions_min", 3)),
            constraints_min=int(frontmatter.get("constraints_min", 3)),
            roles_expected=list(frontmatter.get("roles_expected", [])),
        )

    def _limits_payload(self, limits: RequirementsLimits) -> Dict:
        return {
            "requirements_min": limits.req_min,
            "requirements_max": limits.req_max,
            "assumptions_min": limits.assumptions_min,
            "constraints_min": limits.constraints_min,
            "roles_expected": limits.roles_expected,
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
        return (
            template.replace("{{REQ_MIN}}", str(limits.req_min))
            .replace("{{REQ_MAX}}", str(limits.req_max) if limits.req_max is not None else "none")
            .replace("{{ASSUMPTIONS_MIN}}", str(limits.assumptions_min))
            .replace("{{CONSTRAINTS_MIN}}", str(limits.constraints_min))
            .replace("{{ROLES_EXPECTED}}", roles)
        )

    def _parse_frontmatter(self, content: str) -> Tuple[Dict, str]:
        if not content.startswith("---"):
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

    def _write_usage(self, path: Path, response: LLMResponse) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            write_json(path, usage)

    def _env(self, key: str, default: str) -> str:
        return os.getenv(key, default)

    def _load_schema(self, name: str) -> Dict:
        return json.loads(read_text(self.schemas_dir / name))
