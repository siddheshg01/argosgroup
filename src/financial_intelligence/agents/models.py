"""Shared request and response objects for financial intelligence agents."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AgentRequest:
    question: str
    context: dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    agent: str
    status: str
    findings: list[dict[str, Any]] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    sources: list[dict[str, Any]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
