"""API tests for report access and approval-gated actions."""
import json

from fastapi.testclient import TestClient

from src.financial_intelligence import api


def client_for(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "OUTPUT", tmp_path)
    monkeypatch.delenv("FINOP_API_TOKEN", raising=False)
    return TestClient(api.app)


def test_dashboard_uses_saved_financial_reports(tmp_path, monkeypatch):
    client = client_for(tmp_path, monkeypatch)
    (tmp_path / "financial_report.json").write_text(json.dumps({
        "financial_summary": {"total_revenue": 4321}, "kpis": {"total_orders": 5},
        "risk_analysis": {"risk_score": 30, "risk_level": "MEDIUM"},
        "trend_analysis": {"monthly": [{"month": "2025-01", "revenue": 4321}]}}), encoding="utf-8")
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    assert response.json()["financial_summary"]["total_revenue"] == 4321
    assert response.json()["monthly_trends"][0]["revenue"] == 4321
    assert response.json()["recommendations"] == []


def test_dashboard_includes_saved_agent_recommendations(tmp_path, monkeypatch):
    client = client_for(tmp_path, monkeypatch)
    recommendation = {"text": "Review the declining category with its owner.", "evidence_refs": ["phase2"]}
    (tmp_path / "agentic_financial_analysis.json").write_text(json.dumps({
        "analysis": {"recommendations": [recommendation]}}), encoding="utf-8")
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    assert response.json()["recommendations"] == [recommendation]


def test_invalid_metric_and_question_are_rejected(tmp_path, monkeypatch):
    client = client_for(tmp_path, monkeypatch)
    assert client.get("/api/root-cause?metric=profit").status_code == 422
    assert client.post("/api/agents/query", json={"question": "x"}).status_code == 422


def test_authentication_gate_is_server_configurable(tmp_path, monkeypatch):
    client = client_for(tmp_path, monkeypatch)
    monkeypatch.setenv("FINOP_API_TOKEN", "server-only-test-token")
    assert client.get("/api/actions").status_code == 401
    assert client.get("/api/dashboard").status_code == 401
    assert client.get("/api/actions", headers={"Authorization": "Bearer server-only-test-token"}).status_code == 200


def test_approved_action_required_and_execution_actor_is_audited(tmp_path, monkeypatch):
    client = client_for(tmp_path, monkeypatch)
    action = {"action_id": "act-1", "status": "PENDING", "timestamps": {}, "approval": None,
              "execution_attempted_at": None, "recommendation": {}}
    (tmp_path / "action_proposals.json").write_text(json.dumps({"actions": [action]}), encoding="utf-8")
    response = client.post("/api/actions/act-1/execute", json={"actor": "reviewer-1", "confirm": True})
    assert response.status_code == 200
    assert response.json()["status"] == "blocked"
    audit = json.loads((tmp_path / "workflow_audit.json").read_text(encoding="utf-8"))
    assert audit[-1]["actor"] == "reviewer-1"


def test_execution_requires_explicit_confirmation(tmp_path, monkeypatch):
    client = client_for(tmp_path, monkeypatch)
    response = client.post("/api/actions/not-real/execute", json={"actor": "reviewer", "confirm": False})
    assert response.status_code == 400


def test_unknown_report_path_is_not_served(tmp_path, monkeypatch):
    client = client_for(tmp_path, monkeypatch)
    assert client.get("/api/reports/../../.env").status_code == 404


def test_policy_route_calls_existing_rag_service(monkeypatch):
    client = TestClient(api.app)
    expected = {"status": "no_relevant_documents", "source_citations": [], "answer": "No relevant policy found."}

    class FakePipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, query):
            assert query == "refund approval policy"
            return expected

    monkeypatch.setattr("src.financial_intelligence.rag.rag_pipeline.RAGPipeline", FakePipeline)
    response = client.post("/api/policy/ask", json={"question": "refund approval policy"})
    assert response.status_code == 200
    assert response.json() == expected


def test_ai_analyst_parses_gemini_envelope_using_phase4_parser(tmp_path, monkeypatch):
    client = client_for(tmp_path, monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "server-test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    monkeypatch.setattr("src.financial_intelligence.llm_analyst.load_local_env", lambda _path: None)
    for filename, content in {
        "financial_report.json": {"financial_summary": {"total_revenue": 1200}, "risk_analysis": {"risk_score": 30}},
        "root_cause_report.json": {"significant_changes": [], "significant_change_count": 0},
        "forecast_report.json": {"metrics_forecasted": ["revenue"], "forecasts": []},
    }.items():
        (tmp_path / filename).write_text(json.dumps(content), encoding="utf-8")

    sections = {key: [] for key in (
        "executive_summary", "performance_analysis", "root_cause_analysis", "forecast_analysis",
        "trends", "risks", "opportunities", "attention_items", "recommendations", "data_limitations")}
    sections["forecast_analysis"] = [{"text": "Review the reported revenue forecast.", "classification": "forecast", "evidence_ids": ["phase3_forecasts"]}]
    sections["risks"] = [{"text": "Monitor the reported business risk factors.", "classification": "fact", "evidence_ids": ["phase1_risk"]}]
    envelope = {"candidates": [{"content": {"parts": [{"text": json.dumps(sections)}]}}]}

    def fake_gemini(prompt, api_key, model, timeout):
        assert "What is the expected revenue for the next 6 months, and what risks should management monitor?" in prompt
        assert api_key == "server-test-key"
        assert model == "gemini-test-model"
        return envelope

    monkeypatch.setattr("src.financial_intelligence.llm_analyst.call_gemini", fake_gemini)
    response = client.post("/api/analyst/ask", json={
        "question": "What is the expected revenue for the next 6 months, and what risks should management monitor?"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert "reported revenue forecast" in body["answer"]
    assert "business risk factors" in body["answer"]
    assert body["analysis"]["forecast_analysis"][0]["classification"] == "forecast"
    assert body["analysis"]["risks"][0]["evidence_ids"] == ["phase1_risk"]


def test_ai_analyst_uses_deterministic_reports_without_gemini(tmp_path, monkeypatch):
    client = client_for(tmp_path, monkeypatch)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr("src.financial_intelligence.llm_analyst.load_local_env", lambda _path: None)
    expected = {"request_id": "offline-1", "question": "Summarize financial performance", "status": "complete",
                "analysis": {"facts": [{"text": "Reported revenue is ₹1,200."}]}, "final_answer": "Reported revenue is ₹1,200.",
                "limitations": []}
    monkeypatch.setattr("src.financial_intelligence.agents.orchestrator.run_agent_query", lambda question, output: expected.copy())

    response = client.post("/api/analyst/ask", json={"question": "Summarize financial performance"})

    assert response.status_code == 200
    assert response.json()["status"] == "answered"
    assert response.json()["answer"] == "Reported revenue is ₹1,200."
    assert "deterministic report-based analysis" in response.json()["limitations"][-1]


def test_ai_analyst_empty_valid_gemini_response_has_explanatory_answer(tmp_path, monkeypatch):
    client = client_for(tmp_path, monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "server-test-key")
    monkeypatch.setattr("src.financial_intelligence.llm_analyst.load_local_env", lambda _path: None)
    for name in ("financial_report.json", "root_cause_report.json", "forecast_report.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    empty = {key: [] for key in (
        "executive_summary", "performance_analysis", "root_cause_analysis", "forecast_analysis",
        "trends", "risks", "opportunities", "attention_items", "recommendations", "data_limitations")}
    envelope = {"candidates": [{"content": {"parts": [{"text": json.dumps(empty)}]}}]}
    monkeypatch.setattr("src.financial_intelligence.llm_analyst.call_gemini", lambda *args: envelope)
    response = client.post("/api/analyst/ask", json={"question": "What financial risks should be reviewed?"})
    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_evidence"
    assert response.json()["answer"].strip()


def test_ai_analyst_response_schema_failure_is_logged_not_leaked(tmp_path, monkeypatch, caplog):
    client = client_for(tmp_path, monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "server-test-key")
    monkeypatch.setattr("src.financial_intelligence.llm_analyst.load_local_env", lambda _path: None)
    for name in ("financial_report.json", "root_cause_report.json", "forecast_report.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    monkeypatch.setattr("src.financial_intelligence.llm_analyst.call_gemini", lambda *args: {"candidates": []})
    response = client.post("/api/analyst/ask", json={"question": "summarize financial performance"})
    assert response.status_code == 502
    assert "server-test-key" not in response.text
    assert "AI analyst request" in caplog.text
    assert "Traceback" in caplog.text
