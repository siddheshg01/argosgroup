"""Phase 5 RAG adapter that preserves retrieved policy citations."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from ..rag.rag_pipeline import RAGPipeline
from .models import AgentRequest, AgentResponse


class PolicyRAGAgent:
    name = "policy_rag_agent"
    def __init__(self, rag_pipeline: RAGPipeline | None = None, phase4_path: str | Path = "output/llm_financial_analysis.json"):
        self.rag_pipeline = rag_pipeline or RAGPipeline()
        self.phase4_path = Path(phase4_path)

    def run(self, request: AgentRequest) -> AgentResponse:
        financial_analysis = None
        if self.phase4_path.is_file():
            try:
                financial_analysis = json.loads(self.phase4_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                financial_analysis = None
        report = self.rag_pipeline.run(query=request.question, financial_analysis=financial_analysis)
        citations = report.get("source_citations", [])
        findings = [{"classification": "policy", "text": claim.get("text", ""),
                     "source_chunk_ids": claim.get("source_chunk_ids", []), "evidence_refs": [f"policy:{x}" for x in claim.get("source_chunk_ids", [])]}
                    for claim in report.get("claims", [])]
        if not findings and report.get("answer"):
            findings = [{"classification": "policy", "text": report["answer"], "evidence_refs": [f"policy:{c['chunk_id']}" for c in citations]}]
        status = "success" if report.get("status") in {"answered", "no_relevant_documents", "insufficient_policy_evidence"} else "error"
        limitations = list(report.get("limitations", []))
        if report.get("status") == "no_relevant_documents":
            limitations.append("No relevant policy passage was retrieved; no policy conclusion is asserted.")
        return AgentResponse(self.name, status, findings,
                             {"answer": report.get("answer"), "confidence": report.get("confidence"),
                              "retrieved_documents": report.get("retrieved_documents", []),
                              "relevance_scores": report.get("relevance_scores", []),
                              "claims": report.get("claims", []), "status": report.get("status")},
                             [{"source": c.get("source"), "page": c.get("page"), "section": c.get("section"),
                               "chunk_id": c.get("chunk_id"), "relevance_score": c.get("relevance_score")} for c in citations],
                             limitations)
