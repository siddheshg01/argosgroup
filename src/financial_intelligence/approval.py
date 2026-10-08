"""Mandatory human approval transitions for workflow action proposals."""
from __future__ import annotations
from enum import Enum
from typing import Any
from .workflow_store import WorkflowStore, utc_now


class ActionStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"


class ApprovalEngine:
    def __init__(self, store: WorkflowStore):
        self.store = store

    def decide(self, action_id: str, decision: str, approver: str, reason: str = "") -> dict[str, Any]:
        if not approver or not approver.strip():
            raise ValueError("A human approver name is required")
        decision = decision.strip().upper()
        if decision not in {"APPROVED", "REJECTED"}:
            raise ValueError("decision must be APPROVED or REJECTED")
        actions = self.store.load_actions()
        action = next((item for item in actions if item["action_id"] == action_id), None)
        if action is None:
            raise KeyError(f"Unknown action_id: {action_id}")
        if action["status"] != ActionStatus.PENDING.value:
            raise ValueError(f"Only PENDING actions can be approved or rejected (current: {action['status']})")
        now = utc_now()
        action["status"] = decision
        action["approval"] = {"decision": decision, "approver": approver.strip(), "reason": reason, "timestamp": now}
        action["timestamps"]["approved_at" if decision == "APPROVED" else "rejected_at"] = now
        self.store.save_actions(actions)
        self.store.append_audit({"event": decision.lower(), "action_id": action_id, "recommendation": action.get("recommendation"),
                                 "evidence": action.get("evidence"), "policy_sources": action.get("policy_sources"),
                                 "approver": approver.strip(), "reason": reason, "timestamp": now})
        return action
