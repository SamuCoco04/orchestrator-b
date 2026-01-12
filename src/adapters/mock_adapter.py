from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict

from .llm_base import LLMAdapter, LLMResponse


@dataclass
class MockAdapter(LLMAdapter):
    scenario: str = "default"

    def complete(self, prompt: str) -> LLMResponse:
        payload = self._build_payload(prompt)
        return LLMResponse(raw_text=json.dumps(payload))

    def _build_payload(self, prompt: str) -> Dict:
        if "architecture_candidates" in prompt:
            return {
                "candidates": [
                    {
                        "id": "A",
                        "name": "Modular Pipeline",
                        "summary": "Composable stages with strict contracts.",
                        "pros": ["Deterministic", "Extensible"],
                        "cons": ["More files"],
                        "risks": ["Overhead"]
                    },
                    {
                        "id": "B",
                        "name": "Single Script",
                        "summary": "Monolithic flow in one file.",
                        "pros": ["Simple"],
                        "cons": ["Harder to extend"],
                        "risks": ["Scaling issues"]
                    }
                ]
            }
        if "scoring_result" in prompt:
            return {
                "scores": [
                    {"candidate_id": "A", "total_score": 0.86, "breakdown": {"feasibility": 0.9}},
                    {"candidate_id": "B", "total_score": 0.6, "breakdown": {"feasibility": 0.6}}
                ],
                "selected_id": "A",
                "rationale": "Best balance across rubric criteria."
            }
        if "audit_report" in prompt:
            return {
                "summary": "Mock audit clean.",
                "findings": []
            }
        if "adr" in prompt:
            return {
                "adr_id": "ADR-01",
                "title": "Adopt modular pipeline",
                "status": "Accepted",
                "context": "Need deterministic orchestration with strict contracts.",
                "decision": "Use modular pipeline with adapters and artifacts.",
                "consequences": ["Clear separation of stages", "More files to maintain"]
            }
        return {
            "requirements": [
                {"id": "REQ-1", "text": "Run pipeline in mock mode", "priority": "must"},
                {"id": "REQ-2", "text": "Validate JSON outputs", "priority": "must"}
            ],
            "assumptions": ["No external API keys"],
            "constraints": ["Deterministic outputs"]
        }
