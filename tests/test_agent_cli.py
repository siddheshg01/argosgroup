import json

from src.financial_intelligence.agents import cli
from src.financial_intelligence.agents.models import AgentResponse
from src.financial_intelligence.agents.orchestrator import OrchestratorAgent


def test_cli_displays_returned_answer_without_running_agents_again(monkeypatch, capsys):
    result = {
        "question": "What's the revenue for the last 6 months?",
        "selected_agents": ["financial_analyst"],
        "status": "complete",
        "final_answer": "The last six observed months total ₹8.2 million in revenue.",
        "analysis": {
            "recommendations": [{"text": "Review the monthly trend.", "evidence_refs": ["phase1"]}],
        },
        "sources": [{"source": "financial_report.json"}],
        "limitations": ["Profit and cost are unavailable."],
    }
    calls = []

    def fake_run(question):
        calls.append(question)
        return result

    monkeypatch.setattr(cli, "run_agent_query", fake_run)
    monkeypatch.setattr("sys.argv", ["finop-agents", "What's", "the", "revenue", "for", "the", "last", "6", "months?"])

    cli.main()

    output = capsys.readouterr().out
    assert calls == ["What's the revenue for the last 6 months?"]
    assert "FINOP AI ANSWER" in output
    assert "The last six observed months total ₹8.2 million in revenue." in output
    assert "Recommendations:" in output and "Review the monthly trend." in output
    assert "Sources:" in output and "financial_report.json" in output
    assert "Limitations:" in output and "Profit and cost are unavailable." in output
    assert "output/agentic_financial_analysis.json" in output


def test_cli_handles_missing_answer_gracefully(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_agent_query", lambda question: {
        "question": question, "selected_agents": [], "status": "partial", "analysis": {},
    })
    monkeypatch.setattr("sys.argv", ["finop-agents", "a question"])

    cli.main()

    assert "No natural-language answer was returned by the selected agents." in capsys.readouterr().out


def test_orchestrator_persists_rendered_answer_in_json(tmp_path):
    class AnswerAgent:
        def run(self, request):
            return AgentResponse("financial_analyst", "success", [{
                "classification": "fact", "text": "Last observed month revenue was 1,332,515.68.",
                "evidence_refs": ["phase1"],
            }])

    result = OrchestratorAgent({"financial_analyst": AnswerAgent()}).run("revenue", tmp_path)
    saved = json.loads((tmp_path / "agentic_financial_analysis.json").read_text(encoding="utf-8"))

    assert result["final_answer"] == saved["final_answer"]
    assert "Last observed month revenue was 1,332,515.68." in saved["final_answer"]


def test_last_six_months_question_uses_returned_monthly_evidence_and_persists_answer(tmp_path):
    class MonthlyEvidenceAgent:
        def run(self, request):
            months = ["2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12"]
            return AgentResponse("financial_analyst", "success", [{
                "classification": "fact", "text": "Loaded financial evidence.", "evidence_refs": ["phase1"],
            }], {"trend_analysis": {"monthly": [
                {"_month": month, "revenue": index} for index, month in enumerate(months, 1)
            ]}})

    result = OrchestratorAgent({"financial_analyst": MonthlyEvidenceAgent()}).run("Whats Last 6 Months revenue", tmp_path)
    saved = json.loads((tmp_path / "agentic_financial_analysis.json").read_text(encoding="utf-8"))

    assert "2024-07 to 2024-12" in result["final_answer"]
    assert "Total: ₹21.00" in result["final_answer"]
    assert saved["final_answer"] == result["final_answer"]
