import json
from pathlib import Path

import pytest

from src.financial_intelligence.agents.models import AgentRequest, AgentResponse
from src.financial_intelligence.agents.financial_analyst import FinancialAnalystAgent
from src.financial_intelligence.agents.root_cause_agent import RootCauseAgent
from src.financial_intelligence.agents.forecast_agent import ForecastAgent
from src.financial_intelligence.agents.policy_rag_agent import PolicyRAGAgent
from src.financial_intelligence.agents.recommendation_agent import (
    RecommendationAgent, build_recommendation_context, validate_recommendations,
)
from src.financial_intelligence.agents.orchestrator import OrchestratorAgent, plan_agents


class FakeAgent:
    def __init__(self, name, status="success", failure=None):
        self.name, self.status, self.failure = name, status, failure
        self.calls = []
    def run(self, request):
        self.calls.append(request)
        if self.failure: raise self.failure
        if self.name == "recommendation_agent":
            return AgentResponse(self.name, self.status, [{"classification": "recommendation", "text": "Review the cited revenue changes.", "evidence_refs": ["phase1"]}])
        return AgentResponse(self.name, self.status, [{"classification": "fact", "text": self.name, "evidence_refs": [self.name]}], {"example": 10})


def test_structured_agent_request_and_response():
    req = AgentRequest("question", {"k": 1}, "req-1")
    res = AgentResponse("test", "success", [{"classification": "fact"}], {"x": 1}, [{"source": "a"}], [])
    assert req.question == "question" and req.context["k"] == 1
    assert res.to_dict()["status"] == "success"
    assert set(res.to_dict()) == {"agent", "status", "findings", "evidence", "sources", "limitations"}


def test_phase1_financial_agent_reuses_compact_json_report(tmp_path):
    path = tmp_path / "p1.json"
    path.write_text(json.dumps({"metadata": {"unavailable_fields": ["profit", "cost"]},
        "financial_summary": {"revenue": 20}, "kpis": {"orders": 2}, "data_quality": {},
        "trend_analysis": {}, "location_analysis": [], "regional_analysis": [], "product_analysis": [],
        "category_analysis": [], "concentration_analysis": {}, "risk_analysis": {}}), encoding="utf-8")
    response = FinancialAnalystAgent(path).run(AgentRequest("revenue"))
    assert response.status == "success"
    assert response.evidence["financial_summary"]["revenue"] == 20
    assert "revenue is ₹20.00" in response.findings[0]["text"]
    assert "profit" not in response.findings[0]["text"].casefold()
    assert "Profit and cost are unavailable" in response.limitations[0]


def test_root_cause_agent_filters_metric_and_disclaims_causation(tmp_path):
    path = tmp_path / "p2.json"
    changes = [{"metric": "revenue", "period": "2026-01", "percentage_change": -12.0,
                "contributors": {"product": [{"value": "Product A"}], "location": [{"value": "Austin"}]}},
               {"metric": "quantity", "period": "2026-01", "percentage_change": 5.0, "contributors": []}]
    path.write_text(json.dumps({"significant_changes": changes, "metrics_investigated": ["revenue"], "significant_change_count": 2}), encoding="utf-8")
    response = RootCauseAgent(path).run(AgentRequest("why did revenue decline?"))
    assert len(response.findings) == 1
    assert "not proven causes" in response.findings[0]["text"]
    assert response.findings[0]["contributors"]["product"][0]["value"] == "Product A"


def test_forecast_agent_returns_only_forecast_values_and_horizons(tmp_path):
    path = tmp_path / "p3.json"
    path.write_text(json.dumps({"metrics_forecasted": ["revenue"], "forecasts": [{"metric": "revenue", "status": "forecasted",
        "selected_model": {"name": "naive"}, "validation_metrics": {"mae": 2}, "forecasts_by_horizon": {"3_months": [10]},
        "trend": {"direction": "increasing"}, "assumptions": []}]}), encoding="utf-8")
    response = ForecastAgent(path).run(AgentRequest("forecast revenue"))
    assert response.status == "success"
    assert response.findings[0]["classification"] == "forecast"
    assert response.evidence["forecasts"][0]["forecasts_by_horizon"]["3_months"] == [10]


def test_agent_routing_and_recommendation_dependencies():
    selected, dependencies = plan_agents("Recommend next steps for revenue decline with forecast and policy implications")
    assert selected == ["financial_analyst", "root_cause_agent", "forecast_agent", "policy_rag_agent", "recommendation_agent"]
    assert "must run first" in dependencies["recommendation_agent"]
    assert plan_agents("Forecast revenue next quarter")[0] == ["financial_analyst", "forecast_agent"]


def test_orchestrator_runs_in_order_once_and_writes_reports(tmp_path):
    names = ["financial_analyst", "root_cause_agent", "forecast_agent", "policy_rag_agent", "recommendation_agent"]
    agents = {name: FakeAgent(name) for name in names}
    result = OrchestratorAgent(agents).run("Recommend next steps for revenue decline with forecast and policy", tmp_path)
    assert result["selected_agents"] == names
    assert [x["agent"] for x in result["trace"]] == names
    assert all(len(agents[name].calls) == 1 for name in names)
    assert result["limits"]["actual_agent_calls"] == 5
    assert result["analysis"]["recommendations"]
    assert (tmp_path / "agentic_financial_analysis.json").exists()
    assert (tmp_path / "agentic_financial_analysis.md").exists()


def test_orchestrator_handles_failure_and_continues(tmp_path):
    names = ["financial_analyst", "root_cause_agent", "forecast_agent", "policy_rag_agent", "recommendation_agent"]
    agents = {name: FakeAgent(name, failure=RuntimeError("simulated")) if name == "root_cause_agent" else FakeAgent(name) for name in names}
    result = OrchestratorAgent(agents).run("Recommend next steps for revenue decline, forecast and policy", tmp_path)
    assert result["status"] == "partial"
    assert result["agent_results"][1]["status"] == "error"
    assert len(result["trace"]) == 5


def test_orchestrator_enforces_loop_protection(monkeypatch, tmp_path):
    monkeypatch.setattr("src.financial_intelligence.agents.orchestrator.plan_agents",
                        lambda q: (["financial_analyst", "financial_analyst", "root_cause_agent"], {}))
    financial = FakeAgent("financial_analyst")
    root = FakeAgent("root_cause_agent")
    result = OrchestratorAgent({"financial_analyst": financial, "root_cause_agent": root}, max_agent_calls=5).run("query", tmp_path)
    assert len(financial.calls) == 1
    assert len(root.calls) == 1
    assert any(item["status"] == "blocked" for item in result["trace"])


def test_recommendation_context_separates_sources_and_keeps_rag_citations():
    responses = [AgentResponse("financial_analyst", "success", evidence={"kpis": {"revenue": 500}}),
                 AgentResponse("policy_rag_agent", "success", [{"text": "Approval is required.", "source_chunk_ids": ["abc"], "evidence_refs": ["policy:abc"]}],
                               {"retrieved_documents": [{"id": "abc", "text": "Approval is required."}]},
                               [{"chunk_id": "abc", "source": "policy.pdf", "page": 2}])]
    sections, catalog = build_recommendation_context(responses)
    assert sections["financial_facts"]["evidence"]["kpis"]["revenue"] == 500
    assert "policy:abc" in catalog


def test_recommendation_grounding_and_policy_references():
    catalog = {"phase1": {"revenue": 500}, "policy:chunk1": {"text": "Approval is needed above 500."}}
    result = validate_recommendations({"recommendations": [{"text": "Review approval for the recorded 500 revenue.", "evidence_refs": ["phase1"]}], "limitations": []}, catalog)
    assert result["recommendations"][0]["classification"] == "recommendation"
    with pytest.raises(ValueError, match="Unsupported numeric"):
        validate_recommendations({"recommendations": [{"text": "Escalate 999 items.", "evidence_refs": ["phase1"]}], "limitations": []}, catalog)
    with pytest.raises(ValueError, match="must cite"):
        validate_recommendations({"recommendations": [{"text": "Review this.", "evidence_refs": ["made-up"]}], "limitations": []}, catalog)


def test_recommendation_agent_uses_structured_gemini_result(monkeypatch):
    deps = [AgentResponse("financial_analyst", "success", evidence={"kpis": {"revenue": 500}})]
    monkeypatch.setattr("src.financial_intelligence.agents.recommendation_agent._call_gemini",
        lambda prompt, refs: {"recommendations": [{"text": "Review the revenue trend.", "evidence_refs": ["phase1"]}], "limitations": []})
    response = RecommendationAgent().run(AgentRequest("what should we do", {"agent_responses": deps}))
    assert response.status == "success"
    assert response.findings[0]["classification"] == "recommendation"


def test_policy_agent_preserves_citations_from_rag(tmp_path):
    class FakeRAG:
        def run(self, query, financial_analysis):
            return {"status": "answered", "answer": "Approval is required.", "confidence": "high",
                    "claims": [{"text": "Approval is required.", "source_chunk_ids": ["c1"]}],
                    "retrieved_documents": [{"id": "c1", "text": "Approval is required."}],
                    "relevance_scores": [{"chunk_id": "c1", "score": 0.9}], "limitations": [],
                    "source_citations": [{"chunk_id": "c1", "source": "policy.pdf", "page": 2, "section": "Approval", "relevance_score": 0.9}]}
    response = PolicyRAGAgent(FakeRAG(), tmp_path / "missing.json").run(AgentRequest("what policy applies"))
    assert response.status == "success"
    assert response.sources[0]["page"] == 2
    assert response.findings[0]["evidence_refs"] == ["policy:c1"]
