from pathlib import Path

import pytest

from src.pipeline_requirements import RequirementsFormatError, RequirementsLimits, RequirementsPipeline


def _make_limits(coverage_prefix_mode: bool = False) -> RequirementsLimits:
    return RequirementsLimits(
        req_min=1,
        req_max=None,
        assumptions_min=3,
        constraints_min=3,
        min_student_reqs=0,
        min_coordinator_reqs=0,
        min_admin_reqs=0,
        min_domain_keyword_hits=0,
        roles_expected=[],
        coverage_areas=["Area A", "Area B"],
        coverage_keywords={},
        min_per_area=1,
        coverage_prefix_mode=coverage_prefix_mode,
        seed_requirements=[],
        requested_artifacts=["requirements"],
        artifact_token_budgets={},
        lead_token_budgets={},
        apply_token_budgets={},
    )


def _make_pipeline() -> RequirementsPipeline:
    base_dir = Path(__file__).resolve().parents[1]
    return RequirementsPipeline("mock", base_dir)


def test_missing_required_actions() -> None:
    pipeline = _make_pipeline()
    required = ["Add coverage", "Fix ambiguity"]
    applied = [{"action": "Add coverage", "evidence": "REQ-001"}]
    missing = pipeline._missing_required_actions(required, applied)
    assert missing == ["Fix ambiguity"]


def test_coverage_prefix_counts_unmapped() -> None:
    pipeline = _make_pipeline()
    limits = _make_limits(coverage_prefix_mode=True)
    payload = {
        "requirements": [
            {"id": "REQ-001", "text": "[Area A] Sample requirement.", "priority": "must"},
            {"id": "REQ-002", "text": "No prefix here.", "priority": "should"},
        ]
    }
    counts = pipeline._coverage_counts(payload, limits)
    assert counts["Area A"] == 1
    assert counts["Area B"] == 0
    assert counts["UNMAPPED"] == 1


def test_single_requirement_object_format_error() -> None:
    pipeline = _make_pipeline()
    raw = '{"id":"REQ-001","text":"Test requirement","priority":"must"}'
    with pytest.raises(RequirementsFormatError):
        pipeline._extract_wrapped_json(
            raw,
            "FINAL_REQUIREMENTS_JSON",
            {"requirements", "assumptions", "constraints"},
            context="requirements_apply",
        )
