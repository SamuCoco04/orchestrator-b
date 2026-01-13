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
        if isinstance(payload, str):
            return LLMResponse(raw_text=payload)
        return LLMResponse(raw_text=json.dumps(payload))

    def generate(self, prompt: str) -> str:
        return self.complete(prompt).raw_text

    def _build_payload(self, prompt: str) -> Dict:
        if "requirements_chatgpt_lead" in prompt:
            review = {
                "accepted": ["Add Erasmus Coordinator role"],
                "rejected": [],
                "issues": ["Epic requirements need splitting"],
                "missing": ["Define compliance checkpoints"],
                "rationale": ["Role clarity reduces process risk"],
            }
            requirements = {
                "requirements": [
                    {"id": "REQ-1", "text": "Add Erasmus Coordinator role.", "priority": "must"},
                    {"id": "REQ-2", "text": "Track exchange deadlines.", "priority": "must"},
                    {"id": "REQ-3", "text": "Provide student advising workflow.", "priority": "should"},
                ],
                "assumptions": [],
                "constraints": [],
            }
            return (
                "REVIEW_JSON:\n"
                f"{json.dumps(review)}\n"
                "REQUIREMENTS_JSON:\n"
                f"{json.dumps(requirements)}"
            )
        if "requirements_apply_chatgpt" in prompt:
            final_requirements = {
                "requirements": [
                    {"id": "REQ-2", "text": "Track exchange deadlines.", "priority": "must"},
                    {"id": "REQ-3", "text": "Provide student advising workflow.", "priority": "should"},
                    {"id": "REQ-4", "text": "Define compliance checkpoints.", "priority": "must"},
                    {"id": "REQ-5", "text": "Notify students of deadline changes.", "priority": "should"},
                    {"id": "REQ-6", "text": "Maintain audit trail for approvals.", "priority": "could"},
                    {"id": "REQ-8", "text": "Offer Erasmus Coordinator dashboard.", "priority": "should"},
                    {"id": "REQ-9", "text": "Assign coordinator backup role.", "priority": "could"},
                    {"id": "REQ-10", "text": "Integrate calendar sync for deadlines.", "priority": "could"},
                    {"id": "REQ-11", "text": "Capture eligibility status per student.", "priority": "must"},
                    {"id": "REQ-13", "text": "Support deadline escalation workflow.", "priority": "should"},
                    {"id": "REQ-15", "text": "Generate compliance reports.", "priority": "must"},
                    {"id": "REQ-16", "text": "Store partner university requirements.", "priority": "should"},
                    {"id": "REQ-17", "text": "Provide multilingual notices.", "priority": "could"},
                    {"id": "REQ-18", "text": "Track application milestones.", "priority": "must"},
                    {"id": "REQ-19", "text": "Notify stakeholders of status changes.", "priority": "should"},
                    {"id": "REQ-20", "text": "Support exception approvals.", "priority": "could"},
                    {"id": "REQ-21", "text": "Track outgoing and incoming exchanges.", "priority": "must"},
                    {"id": "REQ-22", "text": "Maintain contact history.", "priority": "could"},
                    {"id": "REQ-23", "text": "Provide SLA metrics.", "priority": "could"},
                    {"id": "REQ-24", "text": "Capture visa documentation status.", "priority": "must"},
                    {"id": "REQ-25", "text": "Validate partner agreements.", "priority": "should"},
                    {"id": "REQ-26", "text": "Allow deadline override approvals.", "priority": "must"},
                    {"id": "REQ-27", "text": "Support bulk deadline updates.", "priority": "should"},
                    {"id": "REQ-28", "text": "Archive completed exchange cases.", "priority": "could"},
                    {"id": "REQ-29", "text": "Generate onboarding checklists.", "priority": "should"},
                    {"id": "REQ-30", "text": "Provide coordinator task queue.", "priority": "must"},
                    {"id": "REQ-31", "text": "Sync with student records system.", "priority": "must"},
                    {"id": "REQ-32", "text": "Support compliance reminders.", "priority": "should"},
                ],
                "assumptions": [],
                "constraints": [],
            }
            changelog = {
                "splits": [
                    {"from": "REQ-1", "to": ["REQ-2", "REQ-4"], "note": "Split epic into actionable items"}
                ],
                "replacements": [],
                "added": ["REQ-4", "REQ-5", "REQ-6"],
                "removed": ["REQ-1"],
            }
            return (
                "FINAL_REQUIREMENTS_JSON:\n"
                f"{json.dumps(final_requirements)}\n"
                "CHANGELOG_JSON:\n"
                f"{json.dumps(changelog)}"
            )
        if "requirements_gemini_cross_review" in prompt:
            return {
                "agreements": ["Splitting epic requirements is necessary.", "Add compliance checkpoints."],
                "disagreements": [],
                "suggestions": ["Prioritize deadline tracking as must-have."],
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
        if "requirements_chatgpt_review" in prompt:
            return {
                "accepted": ["Add Erasmus Coordinator role"],
                "rejected": [],
                "issues": ["Epic requirements need splitting"],
                "missing": ["Define compliance checkpoints"],
                "rationale": ["Role clarity reduces process risk"],
            }
        if "requirements_chatgpt_requirements" in prompt:
            return {
                "requirements": [
                    {"id": "REQ-1", "text": "Add Erasmus Coordinator role.", "priority": "must"},
                    {"id": "REQ-7", "text": "Track exchange deadlines.", "priority": "must"},
                    {"id": "REQ-12", "text": "Provide student advising workflow.", "priority": "should"},
                    {"id": "REQ-14", "text": "Define compliance checkpoints.", "priority": "must"},
                ],
                "assumptions": [],
                "constraints": [],
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
