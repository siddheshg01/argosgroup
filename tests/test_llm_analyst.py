import json
import src.financial_intelligence.llm_analyst as analyst_module
from src.financial_intelligence.llm_analyst import (
    LLMFinancialAnalyst, parse_response, validate_analysis,
)
from src.financial_intelligence.prompt_builder import SECTIONS, build_analysis_context, build_prompt


def sample_reports():
    p1 = {"financial_summary": {"total_revenue": 1200}, "data_quality": {"data_quality_score": 98},
          "kpis": {"profit_status": "not_available_in_source_dataset"}, "risk_analysis": {},
          "concentration_analysis": {}, "trend_analysis": {}}
    p2 = {"significant_changes": [{"metric": "revenue", "percentage_change": -10,
          "contributors": [{"name": "P1", "contribution_pct": 50}]}], "metrics_investigated": ["revenue"]}
    p3 = {"forecasts": [{"metric": "revenue", "status": "forecasted", "forecast_values": [1300]}]}
    return p1, p2, p3


def test_context_and_prompt_use_phase1_to_3_reports_only():
    context = build_analysis_context(*sample_reports())
    prompt = build_prompt(context)
    assert "phase1" in prompt and "phase2" in prompt and "phase3" in prompt
    assert "causation" in prompt.lower() and "profit and cost are unavailable" in prompt.lower()
    assert "raw_transactions" not in prompt


def test_response_parsing_handles_gemini_envelope_and_bad_json():
    assert parse_response({"candidates": [{"content": {"parts": [{"text": "{\"x\": 1}"}]}}]}) == {"x": 1}
    try:
        parse_response("not json")
        assert False
    except ValueError:
        pass


def valid_payload(context):
    payload = {s: [] for s in SECTIONS}
    payload["performance_analysis"] = [{"text": "Revenue was 1200.", "classification": "fact",
                                         "evidence_ids": ["phase1_financial_summary"]}]
    return payload


def test_grounding_accepts_cited_value_and_rejects_fabricated_value():
    context = build_analysis_context(*sample_reports())
    assert validate_analysis(valid_payload(context), context)["performance_analysis"]
    payload = valid_payload(context); payload["performance_analysis"][0]["text"] = "Revenue was 987654."
    try:
        validate_analysis(payload, context)
        assert False
    except ValueError as exc:
        assert "Unsupported numeric claim" in str(exc)


def test_unknown_citation_is_rejected():
    context = build_analysis_context(*sample_reports())
    payload = valid_payload(context); payload["performance_analysis"][0]["evidence_ids"] = ["made_up"]
    try:
        validate_analysis(payload, context)
        assert False
    except ValueError as exc:
        assert "Unknown evidence" in str(exc)


def test_forecast_model_order_parameters_are_not_read_as_thousands():
    p1, p2, p3 = sample_reports()
    p3["forecasts"][0]["selected_model"] = {"name": "sarima_1_0_0_12"}
    context = build_analysis_context(p1, p2, p3)
    payload = {section: [] for section in SECTIONS}
    payload["forecast_analysis"] = [{"text": "Selected model: SARIMA(1, 0, 0, 12).", "classification": "fact",
                                      "evidence_ids": ["phase3_forecasts"]}]
    assert validate_analysis(payload, context)["forecast_analysis"]


def test_missing_key_writes_json_and_markdown_fallback(tmp_path, monkeypatch):
    p1, p2, p3 = sample_reports()
    for filename, value in zip(("financial_report.json", "root_cause_report.json", "forecast_report.json"), (p1, p2, p3)):
        (tmp_path / filename).write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False); monkeypatch.delenv("LLM_API_KEY", raising=False)
    report = LLMFinancialAnalyst(tmp_path, tmp_path / "no-env").run()
    assert report["status"] == "unavailable"
    assert (tmp_path / "llm_financial_analysis.json").is_file()
    assert (tmp_path / "llm_financial_analysis.md").is_file()
    assert "profit and cost" in report["data_limitations"][0]["text"] or "GEMINI_API_KEY" in report["error"]


def test_phase_report_missing_is_handled_and_reports_still_written(tmp_path):
    report = LLMFinancialAnalyst(tmp_path, tmp_path / "no-env").run()
    assert report["status"] == "error"
    assert "missing" in report["error"]
    assert (tmp_path / "llm_financial_analysis.json").exists()


def test_gemini_api_failure_is_handled_without_exposing_key(tmp_path, monkeypatch):
    p1, p2, p3 = sample_reports()
    for filename, value in zip(("financial_report.json", "root_cause_report.json", "forecast_report.json"), (p1, p2, p3)):
        (tmp_path / filename).write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "test-secret-value")
    monkeypatch.setattr(analyst_module, "call_gemini", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Gemini API returned HTTP 503")))
    report = LLMFinancialAnalyst(tmp_path, tmp_path / "no-env").run()
    assert report["status"] == "error"
    assert "503" in report["error"]
    assert "test-secret-value" not in json.dumps(report)


def test_invalid_generated_response_is_reported(tmp_path, monkeypatch):
    p1, p2, p3 = sample_reports()
    for filename, value in zip(("financial_report.json", "root_cause_report.json", "forecast_report.json"), (p1, p2, p3)):
        (tmp_path / filename).write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(analyst_module, "call_gemini", lambda *args, **kwargs: {"candidates": [{"content": {"parts": [{"text": "not json"}]}}]})
    report = LLMFinancialAnalyst(tmp_path, tmp_path / "no-env").run()
    assert report["status"] == "error"
    assert "valid JSON" in report["error"]


def test_quota_error_uses_configured_lower_cost_fallback(tmp_path, monkeypatch):
    p1, p2, p3 = sample_reports()
    for filename, value in zip(("financial_report.json", "root_cause_report.json", "forecast_report.json"), (p1, p2, p3)):
        (tmp_path / filename).write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.1-pro-preview")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "gemini-3.1-flash-lite")
    attempted = []
    def fake_call(prompt, key, model, *args):
        attempted.append(model)
        if model == "gemini-3.1-pro-preview":
            raise RuntimeError("Gemini API returned HTTP 429 quota")
        return {section: [] for section in SECTIONS}
    monkeypatch.setattr(analyst_module, "call_gemini", fake_call)
    report = LLMFinancialAnalyst(tmp_path, tmp_path / "no-env").run()
    assert report["status"] == "generated"
    assert report["model"] == "gemini-3.1-flash-lite"
    assert attempted == ["gemini-3.1-pro-preview", "gemini-3.1-flash-lite"]
