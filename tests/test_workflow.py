import json
import smtplib
from pathlib import Path

import pytest

from src.financial_intelligence.actions import ActionType
from src.financial_intelligence.approval import ActionStatus, ApprovalEngine
from src.financial_intelligence.workflow import ActionEngine, ActionValidator, WorkflowEngine
from src.financial_intelligence.workflow_store import WorkflowStore


def phase6_report():
    return {"status": "complete", "phase_1_to_5_integrated": True,
        "agent_results": [
            {"agent": "financial_analyst", "status": "success", "evidence": {"risk_analysis": {"risk_score": 30}, "financial_summary": {"revenue": 1000}}},
            {"agent": "root_cause_agent", "status": "success", "evidence": {"significant_change_count": 1}},
            {"agent": "forecast_agent", "status": "success", "evidence": {"metrics_forecasted": ["revenue"]}},
            {"agent": "policy_rag_agent", "status": "success", "sources": [{"chunk_id": "c1", "source": "refund_policy.pdf", "page": 2, "section": "Approval"}],
             "evidence": {"retrieved_documents": [{"id": "c1", "text": "Refunds require review approval.", "score": 0.84}]}},
        ],
        "analysis": {"recommendations": [
            {"classification": "recommendation", "text": "Review revenue movement as required by policy.", "evidence_refs": ["phase1", "policy:c1"]},
            {"classification": "recommendation", "text": "Monitor the next revenue trend.", "evidence_refs": ["phase3"]},
        ]}}


def make_engine(tmp_path, report=None):
    report_path = tmp_path / "phase6.json"
    report_path.write_text(json.dumps(report or phase6_report()), encoding="utf-8")
    engine = WorkflowEngine(tmp_path, report_path)
    return engine, report_path


def test_action_proposals_preserve_evidence_policy_and_are_idempotent(tmp_path):
    store = WorkflowStore(tmp_path)
    engine = ActionEngine(store)
    report_path = tmp_path / "phase6.json"
    report_path.write_text(json.dumps(phase6_report()), encoding="utf-8")
    first = engine.propose_from_phase6(report_path)
    again = engine.propose_from_phase6(report_path)
    assert len(first["created_action_ids"]) == 2
    assert len(again["reused_action_ids"]) == 2
    action = store.load_actions()[0]
    assert action["status"] == "PENDING"
    assert action["policy_sources"][0]["source"] == "refund_policy.pdf"
    assert action["policy_sources"][0]["text"] == "Refunds require review approval."
    assert {x["reference"] for x in action["evidence"]} == {"phase1", "policy:c1"}


def test_action_validation_rejects_unknown_evidence_and_missing_policy_citation():
    report = phase6_report()
    rec = {"text": "Follow policy now.", "evidence_refs": ["not-real"]}
    _, _, errors = ActionValidator().validate(rec, report)
    assert any("not present" in error for error in errors)
    rec = {"text": "This is required by policy.", "evidence_refs": ["phase1"]}
    _, _, errors = ActionValidator().validate(rec, report)
    assert any("no policy source" in error for error in errors)


def test_human_approval_is_mandatory_and_rejection_blocks_execution(tmp_path):
    engine, report_path = make_engine(tmp_path)
    action_id = engine.actions.propose_from_phase6(report_path)["created_action_ids"][0]
    blocked = engine.execute(action_id)
    assert blocked["status"] == "blocked"
    engine.reject(action_id, "operator-1", "Not appropriate")
    with pytest.raises(ValueError, match="Only PENDING"):
        engine.approve(action_id, "operator-2")
    assert engine.execute(action_id)["status"] == "blocked"


def test_approval_requires_named_human_and_pending_state(tmp_path):
    engine, report_path = make_engine(tmp_path)
    action_id = engine.actions.propose_from_phase6(report_path)["created_action_ids"][0]
    with pytest.raises(ValueError, match="human approver"):
        engine.approve(action_id, " ")
    action = engine.approve(action_id, "operator-1", "Reviewed evidence")
    assert action["status"] == ActionStatus.APPROVED.value
    assert action["approval"]["approver"] == "operator-1"


def test_approved_ticket_executes_once_and_duplicate_is_blocked(tmp_path):
    engine, report_path = make_engine(tmp_path)
    action_id = engine.actions.propose_from_phase6(report_path)["created_action_ids"][0]
    engine.approve(action_id, "operator")
    executed = engine.execute(action_id)
    assert executed["status"] == "EXECUTED"
    duplicate = engine.execute(action_id)
    assert duplicate["status"] == "blocked"
    tickets = json.loads((tmp_path / "actions" / "local_tickets.json").read_text(encoding="utf-8"))
    assert len(tickets) == 1


def test_dry_run_never_performs_side_effect_or_marks_executed(tmp_path):
    engine, report_path = make_engine(tmp_path)
    action_id = engine.actions.propose_from_phase6(report_path)["created_action_ids"][0]
    engine.approve(action_id, "operator")
    result = engine.execute(action_id, dry_run=True)
    assert result["outcome"] == "dry_run_only"
    action = next(x for x in engine.store.load_actions() if x["action_id"] == action_id)
    assert action["status"] == "APPROVED"
    assert action["execution_attempted_at"] is None
    assert not (tmp_path / "actions" / "local_tickets.json").exists()


def test_email_failure_sanitizes_error_and_does_not_leak_secret(tmp_path, monkeypatch):
    engine, report_path = make_engine(tmp_path)
    action_id = engine.actions.propose_from_phase6(report_path, ActionType.EMAIL_NOTIFICATION.value)["created_action_ids"][0]
    engine.approve(action_id, "operator")
    monkeypatch.setenv("SMTP_HOST", "mail.example.test")
    monkeypatch.setenv("SMTP_FROM", "finops@example.test")
    monkeypatch.setenv("SMTP_TO", "manager@example.test")
    class BrokenSMTP:
        def __init__(self, *args, **kwargs): raise RuntimeError("secret-password")
    monkeypatch.setattr(smtplib, "SMTP", BrokenSMTP)
    result = engine.execute(action_id)
    assert result["status"] == "FAILED"
    assert "secret-password" not in json.dumps(result)


def test_report_and_financial_alert_actions_generate_local_outputs(tmp_path):
    engine, report_path = make_engine(tmp_path)
    action_id = engine.actions.propose_from_phase6(report_path, ActionType.MANAGEMENT_REPORT.value)["created_action_ids"][0]
    engine.approve(action_id, "operator")
    result = engine.execute(action_id)
    assert result["status"] == "EXECUTED"
    assert Path(result["execution_result"]["path"]).is_file()

    alert_id = engine.actions.propose_from_phase6(report_path, ActionType.FINANCIAL_ALERT.value)["created_action_ids"][0]
    engine.approve(alert_id, "operator")
    alert = engine.execute(alert_id)
    assert alert["execution_result"]["outcome"] == "alert_recorded"


def test_dry_run_workflow_creates_pending_proposals_and_all_reports(tmp_path):
    engine, report_path = make_engine(tmp_path)
    result = engine.run_dry_run(report_path)
    assert result["status"] == "dry_run_complete"
    assert len(result["pending_approval_ids"]) == 2
    assert result["executed_action_ids"] == []
    for filename in ("action_proposals.json", "workflow_results.json", "workflow_audit.json", "workflow_summary.md"):
        assert (tmp_path / filename).is_file()
    audit = json.loads((tmp_path / "workflow_audit.json").read_text(encoding="utf-8"))
    assert any(event["event"] == "workflow_dry_run" and event["no_side_effects"] for event in audit)


def test_email_action_sends_only_after_approval_with_mock_smtp(tmp_path, monkeypatch):
    engine, report_path = make_engine(tmp_path)
    action_id = engine.actions.propose_from_phase6(report_path, ActionType.EMAIL_NOTIFICATION.value)["created_action_ids"][0]
    engine.approve(action_id, "operator")
    monkeypatch.setenv("SMTP_HOST", "smtp.local")
    monkeypatch.setenv("SMTP_FROM", "finops@example.test")
    monkeypatch.setenv("SMTP_TO", "manager@example.test")
    calls = []
    class FakeSMTP:
        def __init__(self, *args, **kwargs): calls.append("connect")
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def starttls(self): calls.append("tls")
        def send_message(self, message): calls.append(message["To"])
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    result = engine.execute(action_id)
    assert result["execution_result"]["outcome"] == "email_sent"
    assert calls == ["connect", "tls", "manager@example.test"]


def test_workflow_rejects_missing_phase6_source(tmp_path):
    engine = WorkflowEngine(tmp_path)
    with pytest.raises(FileNotFoundError):
        engine.actions.propose_from_phase6(tmp_path / "absent.json")


def test_execution_revalidates_approved_action_against_current_phase6_evidence(tmp_path):
    engine, report_path = make_engine(tmp_path)
    action_id = engine.actions.propose_from_phase6(report_path)["created_action_ids"][0]
    engine.approve(action_id, "operator")
    report_path.write_text(json.dumps({"analysis": {"recommendations": []}}), encoding="utf-8")
    result = engine.execute(action_id)
    assert result["status"] == "blocked"
    assert not (tmp_path / "actions" / "local_tickets.json").exists()


def test_unsupported_action_type_is_rejected(tmp_path):
    engine, report_path = make_engine(tmp_path)
    result = engine.actions.propose_from_phase6(report_path, "wire_transfer")
    assert result["created_action_ids"] == []
    assert result["rejected_recommendations"]
