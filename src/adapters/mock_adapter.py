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

    def generate(self, prompt: str) -> str:
        return self.complete(prompt).raw_text

    def _build_payload(self, prompt: str) -> Dict:
        if "requirements_gemini" in prompt:
            return {
                "roles": [
                    {
                        "name": "Erasmus Coordinator",
                        "responsibilities": [
                            "Set exchange deadlines",
                            "Advise students on application steps",
                            "Coordinate compliance checks",
                        ],
                    }
                ],
                "gaps": ["Missing dedicated exchange coordinator role"],
            }
        if "architecture_candidates" in prompt:
            return {
                "candidates": [
                    {
                        "id": "A",
                        "name": "Modular Pipeline",
                        "summary": "Composable stages with strict contracts.",
                        "pros": ["Deterministic", "Extensible"],
                        "cons": ["More files"],
                        "risks": ["Overhead"],
                    },
                    {
                        "id": "B",
                        "name": "Service-Oriented",
                        "summary": "Independent services with shared artifacts.",
                        "pros": ["Scalable", "Clear boundaries"],
                        "cons": ["Coordination overhead"],
                        "risks": ["Operational complexity"],
                    },
                    {
                        "id": "C",
                        "name": "Workflow Engine",
                        "summary": "Managed workflow orchestrator.",
                        "pros": ["Built-in scheduling"],
                        "cons": ["Vendor lock-in"],
                        "risks": ["Migration cost"],
                    },
                    {
                        "id": "D",
                        "name": "Monolithic Script",
                        "summary": "Single script for all stages.",
                        "pros": ["Simple"],
                        "cons": ["Harder to extend"],
                        "risks": ["Scaling issues"],
                    },
                ]
            }
        if "scoring_result" in prompt:
            return {
                "scores": [
                    {
                        "candidate_id": "A",
                        "total_score": 0.86,
                        "breakdown": {
                            "feasibility": 0.9,
                            "cost": 0.8,
                            "maintainability": 0.9,
                            "scalability": 0.8,
                            "security": 0.85,
                        },
                    },
                    {
                        "candidate_id": "B",
                        "total_score": 0.75,
                        "breakdown": {
                            "feasibility": 0.8,
                            "cost": 0.7,
                            "maintainability": 0.8,
                            "scalability": 0.8,
                            "security": 0.7,
                        },
                    },
                    {
                        "candidate_id": "C",
                        "total_score": 0.62,
                        "breakdown": {
                            "feasibility": 0.6,
                            "cost": 0.5,
                            "maintainability": 0.6,
                            "scalability": 0.7,
                            "security": 0.65,
                        },
                    },
                    {
                        "candidate_id": "D",
                        "total_score": 0.48,
                        "breakdown": {
                            "feasibility": 0.5,
                            "cost": 0.6,
                            "maintainability": 0.4,
                            "scalability": 0.3,
                            "security": 0.6,
                        },
                    },
                ],
                "selected_id": "A",
                "rationale": "Best balance across rubric criteria.",
            }
        if "audit_report" in prompt:
            return {
                "summary": "Mock audit clean.",
                "findings": [],
            }
        if "adr" in prompt:
            return {
                "adr_id": "ADR-01",
                "title": "Adopt modular pipeline",
                "status": "Accepted",
                "context": "Need deterministic orchestration with strict contracts.",
                "decision": "Use modular pipeline with adapters and artifacts.",
                "consequences": ["Clear separation of stages", "More files to maintain"],
            }
        return {
            "requirements": [
                {"id": "REQ-1", "text": "Run pipeline in mock mode", "priority": "must"},
                {"id": "REQ-2", "text": "Validate JSON outputs", "priority": "must"},
            ],
            "assumptions": ["No external API keys"],
            "constraints": ["Deterministic outputs"],
        }
