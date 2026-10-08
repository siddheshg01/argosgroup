"""Common action interface and safe local/SMTP action adapters for Phase 7."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
import json
import os
import smtplib
from email.message import EmailMessage


class ActionType(str, Enum):
    EMAIL_NOTIFICATION = "email_notification"
    FINANCIAL_ALERT = "financial_alert"
    MANAGEMENT_REPORT = "management_report"
    TICKET_TASK = "ticket_task"


@dataclass
class ActionExecution:
    outcome: str
    details: dict[str, Any]


class Action(ABC):
    """Contract implemented by each supported workflow action."""
    action_type: ActionType
    @abstractmethod
    def execute(self, proposal: dict[str, Any], output_dir: Path) -> ActionExecution:
        """Execute an already-approved action and return a safe summary."""


class EmailNotificationAction(Action):
    action_type = ActionType.EMAIL_NOTIFICATION
    def execute(self, proposal: dict[str, Any], output_dir: Path) -> ActionExecution:
        host, sender, recipient = os.getenv("SMTP_HOST"), os.getenv("SMTP_FROM"), os.getenv("SMTP_TO")
        if not host or not sender or not recipient:
            raise RuntimeError("SMTP_HOST, SMTP_FROM, and SMTP_TO must be configured")
        message = EmailMessage()
        message["Subject"] = f"FINOP notification: {proposal['description'][:100]}"
        message["From"], message["To"] = sender, recipient
        message.set_content(_action_body(proposal))
        port = int(os.getenv("SMTP_PORT", "587"))
        username, password = os.getenv("SMTP_USERNAME"), os.getenv("SMTP_PASSWORD")
        use_ssl = os.getenv("SMTP_USE_SSL", "false").strip().lower() in {"1", "true", "yes"}
        try:
            smtp_class = smtplib.SMTP_SSL if use_ssl else smtplib.SMTP
            with smtp_class(host, port, timeout=float(os.getenv("SMTP_TIMEOUT_SECONDS", "20"))) as client:
                if not use_ssl:
                    client.starttls()
                if username and password:
                    client.login(username, password)
                client.send_message(message)
        except Exception as exc:
            # Do not serialize SMTP exception text: libraries/servers may echo details.
            raise RuntimeError(f"SMTP send failed ({type(exc).__name__})") from None
        return ActionExecution("email_sent", {"recipient": recipient, "subject": message["Subject"]})


class FinancialAlertAction(Action):
    action_type = ActionType.FINANCIAL_ALERT
    def execute(self, proposal: dict[str, Any], output_dir: Path) -> ActionExecution:
        path = output_dir / "actions" / "financial_alerts.json"
        _append_json_record(path, {"action_id": proposal["action_id"], "priority": proposal["priority"],
                                   "description": proposal["description"], "evidence": proposal["evidence"],
                                   "created_at": proposal["timestamps"]["created_at"]})
        return ActionExecution("alert_recorded", {"path": str(path)})


class ManagementReportAction(Action):
    action_type = ActionType.MANAGEMENT_REPORT
    def execute(self, proposal: dict[str, Any], output_dir: Path) -> ActionExecution:
        path = output_dir / "actions" / f"management_report_{proposal['action_id']}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        refs = ", ".join(item.get("reference", "") for item in proposal.get("evidence", []))
        policies = ", ".join(item.get("source", "") for item in proposal.get("policy_sources", []))
        path.write_text(f"# Management Action Report\n\n{proposal['description']}\n\nPriority: {proposal['priority']}\n\nEvidence: {refs}\n\nPolicy sources: {policies or 'None cited'}\n", encoding="utf-8")
        return ActionExecution("report_generated", {"path": str(path)})


class TicketTaskAction(Action):
    action_type = ActionType.TICKET_TASK
    def execute(self, proposal: dict[str, Any], output_dir: Path) -> ActionExecution:
        path = output_dir / "actions" / "local_tickets.json"
        ticket = {"ticket_id": "FIN-" + proposal["action_id"][:10].upper(), "action_id": proposal["action_id"],
                  "title": proposal["description"][:120], "description": proposal["description"],
                  "priority": proposal["priority"], "status": "OPEN", "evidence": proposal["evidence"],
                  "policy_sources": proposal["policy_sources"]}
        _append_json_record(path, ticket)
        return ActionExecution("ticket_created", {"ticket_id": ticket["ticket_id"], "path": str(path)})


def _append_json_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    if not any(item.get("action_id") == record.get("action_id") for item in records):
        records.append(record)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def _action_body(proposal: dict[str, Any]) -> str:
    refs = ", ".join(item.get("reference", "") for item in proposal.get("evidence", []))
    policies = ", ".join(item.get("source", "") for item in proposal.get("policy_sources", []))
    return (f"Action: {proposal['description']}\nPriority: {proposal['priority']}\n"
            f"Evidence: {refs}\nPolicy sources: {policies or 'None cited'}\n"
            f"Action ID: {proposal['action_id']}\n")
