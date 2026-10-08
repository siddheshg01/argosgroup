"""Policy/risk validation, proposals, approvals, controlled execution, and dry runs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
from .actions import (ActionType, EmailNotificationAction, FinancialAlertAction,
                      ManagementReportAction, TicketTaskAction)
from .approval import ActionStatus, ApprovalEngine
from .workflow_store import WorkflowStore, utc_now


class ActionValidator:
    """Validate recommendation references and retain exact Phase 6 evidence sources."""
    def validate(self, recommendation: dict[str, Any], phase6: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        refs = recommendation.get("evidence_refs")
        errors: list[str] = []
        if not isinstance(refs, list) or not refs:
            return [], [], ["Recommendation has no evidence references."]
        agent_results = {r.get("agent"): r for r in phase6.get("agent_results", [])}
        available: dict[str, Any] = {}
        for phase, agent_name in (("phase1", "financial_analyst"), ("phase2", "root_cause_agent"), ("phase3", "forecast_agent")):
            if agent_results.get(agent_name, {}).get("status") == "success":
                available[phase] = agent_results[agent_name].get("evidence", {})
        policy_sources: dict[str, dict[str, Any]] = {}
        policy_agent = agent_results.get("policy_rag_agent", {})
        if policy_agent.get("status") == "success":
            retrieved = policy_agent.get("evidence", {}).get("retrieved_documents", [])
            retrieved_by_id = {(item.get("chunk_id") or item.get("id")): item for item in retrieved
                               if item.get("chunk_id") or item.get("id")}
            for source in policy_agent.get("sources", []):
                if source.get("chunk_id"):
                    # Keep citation metadata and the matching retrieved excerpt for auditability.
                    policy_sources[f"policy:{source['chunk_id']}"] = {
                        **source,
                        "text": retrieved_by_id.get(source["chunk_id"], {}).get("text"),
                        "relevance_score": retrieved_by_id.get(source["chunk_id"], {}).get("relevance_score"),
                    }
        evidence: list[dict[str, Any]] = []
        used_policy: list[dict[str, Any]] = []
        for ref in refs:
            if ref in available:
                evidence.append({"reference": ref, "excerpt": available[ref]})
            elif ref in policy_sources:
                source = policy_sources[ref]
                used_policy.append(source)
                evidence.append({"reference": ref, "excerpt": {k: source.get(k) for k in ("source", "page", "section", "chunk_id")}})
            else:
                errors.append(f"Evidence reference is not present in the Phase 6 report: {ref}")
        text = str(recommendation.get("text", ""))
        if not text.strip(): errors.append("Recommendation text is empty.")
        if "policy" in text.casefold() and not used_policy:
            errors.append("Recommendation refers to policy but carries no policy source citation.")
        if not used_policy and not evidence:
            errors.append("No verifiable financial or policy evidence is available.")
        return evidence, used_policy, errors


class ActionEngine:
    """Create deterministic, idempotent PENDING proposals from Phase 6 recommendations."""
    def __init__(self, store: WorkflowStore, validator: ActionValidator | None = None):
        self.store, self.validator = store, validator or ActionValidator()

    def propose_from_phase6(self, report_path: str | Path = "output/agentic_financial_analysis.json",
                            action_type: str | None = None) -> dict[str, Any]:
        path = Path(report_path)
        if not path.is_file(): raise FileNotFoundError(f"Phase 6 report not found: {path}")
        phase6 = json.loads(path.read_text(encoding="utf-8"))
        recommendations = phase6.get("analysis", {}).get("recommendations", [])
        actions = self.store.load_actions()
        existing_by_fingerprint = {item.get("fingerprint"): item for item in actions}
        created, reused, rejected = [], [], []
        for index, recommendation in enumerate(recommendations):
            evidence, policies, errors = self.validator.validate(recommendation, phase6)
            rec_text = str(recommendation.get("text", ""))
            chosen_type = action_type or infer_action_type(rec_text)
            if chosen_type not in {item.value for item in ActionType}:
                rejected.append({"recommendation_index": index, "text": rec_text,
                                 "errors": [f"Unsupported action type: {chosen_type}"]})
                continue
            fingerprint = _fingerprint(chosen_type, rec_text, recommendation.get("evidence_refs", []))
            if fingerprint in existing_by_fingerprint:
                reused.append(existing_by_fingerprint[fingerprint]["action_id"])
                continue
            if errors:
                rejected.append({"recommendation_index": index, "text": rec_text, "errors": errors})
                self.store.append_audit({"event": "proposal_validation_failed", "recommendation": recommendation,
                                         "evidence": evidence, "policy_sources": policies, "errors": errors, "timestamp": utc_now()})
                continue
            priority = _priority_from_evidence(phase6)
            action_id = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
            now = utc_now()
            action = {"action_id": action_id, "type": chosen_type, "description": rec_text, "priority": priority,
                      "evidence": evidence, "policy_sources": policies, "status": ActionStatus.PENDING.value,
                      "recommendation": recommendation, "fingerprint": fingerprint, "approval": None,
                      "execution_attempted_at": None, "execution_result": None,
                      "timestamps": {"created_at": now, "approved_at": None, "rejected_at": None,
                                     "executed_at": None, "failed_at": None}}
            actions.append(action); existing_by_fingerprint[fingerprint] = action; created.append(action_id)
            self.store.append_audit({"event": "proposed", "action_id": action_id, "type": chosen_type,
                                     "recommendation": recommendation, "evidence": evidence,
                                     "policy_sources": policies, "timestamp": now})
        self.store.save_actions(actions)
        result = {"status": "proposals_pending_approval", "created_action_ids": created, "reused_action_ids": reused,
                  "rejected_recommendations": rejected, "actions": actions,
                  "human_approval_required": True, "dry_run": False}
        return result


def infer_action_type(text: str) -> str:
    lower = text.casefold()
    if any(word in lower for word in ("email", "notify", "notification")):
        return ActionType.EMAIL_NOTIFICATION.value
    if any(word in lower for word in ("alert", "monitor", "threshold")):
        return ActionType.FINANCIAL_ALERT.value
    if any(word in lower for word in ("management report", "prepare a report", "report for management")):
        return ActionType.MANAGEMENT_REPORT.value
    return ActionType.TICKET_TASK.value


def _fingerprint(action_type: str, text: str, refs: list[str]) -> str:
    return hashlib.sha256(json.dumps([action_type, text.strip(), sorted(refs)], ensure_ascii=False).encode()).hexdigest()


def _priority_from_evidence(phase6: dict[str, Any]) -> str:
    for result in phase6.get("agent_results", []):
        if result.get("agent") == "financial_analyst":
            score = result.get("evidence", {}).get("risk_analysis", {}).get("risk_score")
            try: score = float(score)
            except (TypeError, ValueError): return "MEDIUM"
            if score >= 76: return "CRITICAL"
            if score >= 51: return "HIGH"
            if score >= 26: return "MEDIUM"
            return "LOW"
    return "MEDIUM"


class WorkflowEngine:
    """Control proposal, human approval, safe action execution, and audit outputs."""
    def __init__(self, output_dir: str | Path = "output",
                 phase6_report_path: str | Path = "output/agentic_financial_analysis.json"):
        self.store = WorkflowStore(output_dir)
        self.phase6_report_path = Path(phase6_report_path)
        self.actions = ActionEngine(self.store)
        self.approvals = ApprovalEngine(self.store)
        self.handlers = {ActionType.EMAIL_NOTIFICATION.value: EmailNotificationAction(),
                         ActionType.FINANCIAL_ALERT.value: FinancialAlertAction(),
                         ActionType.MANAGEMENT_REPORT.value: ManagementReportAction(),
                         ActionType.TICKET_TASK.value: TicketTaskAction()}

    def approve(self, action_id: str, approver: str, reason: str = "") -> dict[str, Any]:
        return self.approvals.decide(action_id, "APPROVED", approver, reason)

    def reject(self, action_id: str, approver: str, reason: str = "") -> dict[str, Any]:
        return self.approvals.decide(action_id, "REJECTED", approver, reason)

    def pending(self) -> list[dict[str, Any]]:
        return [action for action in self.store.load_actions() if action.get("status") == ActionStatus.PENDING.value]

    def execute(self, action_id: str, dry_run: bool = False, actor: str = "") -> dict[str, Any]:
        actions = self.store.load_actions()
        action = next((item for item in actions if item["action_id"] == action_id), None)
        if action is None: raise KeyError(f"Unknown action_id: {action_id}")
        now = utc_now()
        if action["status"] != ActionStatus.APPROVED.value:
            event = {"event": "execution_blocked", "action_id": action_id, "status": action["status"],
                     "actor": actor, "reason": "Human approval is required before execution.", "timestamp": now}
            self.store.append_audit(event)
            return {**event, "status": "blocked"}
        if action.get("execution_attempted_at"):
            event = {"event": "duplicate_execution_blocked", "action_id": action_id,
                     "actor": actor, "reason": "This action already has an execution attempt.", "timestamp": now}
            self.store.append_audit(event)
            return {**event, "status": "blocked"}
        # Revalidate the original evidence immediately before execution. This prevents
        # stale or altered Phase 6 evidence from authorizing an action later.
        try:
            phase6 = json.loads(self.phase6_report_path.read_text(encoding="utf-8"))
            evidence, policies, errors = self.actions.validator.validate(action.get("recommendation", {}), phase6)
        except (OSError, json.JSONDecodeError) as exc:
            evidence, policies, errors = [], [], [f"Phase 6 evidence could not be loaded ({type(exc).__name__})."]
        if errors:
            event = {"event": "execution_blocked", "action_id": action_id,
                     "reason": "Evidence or policy validation failed before execution.",
                     "actor": actor, "validation_errors": errors, "timestamp": now}
            self.store.append_audit(event)
            return {"status": "blocked", **event}
        action["evidence"], action["policy_sources"] = evidence, policies
        self.store.save_actions(actions)
        if dry_run:
            result = {"outcome": "dry_run_only", "message": "No email, ticket, alert, or report side effect was performed."}
            action["execution_result"] = result
            self.store.save_actions(actions)
            self.store.append_audit({"event": "dry_run_execution", "action_id": action_id,
                                     "actor": actor,
                                     "approval": action.get("approval"), "execution_result": result,
                                     "evidence": action.get("evidence"), "policy_sources": action.get("policy_sources"),
                                     "timestamp": now})
            return {"status": "APPROVED", **result}
        # Persist the attempt before performing an external side effect: at-most-once behavior.
        action["execution_attempted_at"] = now
        self.store.save_actions(actions)
        try:
            handler = self.handlers[action["type"]]
            result = handler.execute(action, self.store.output_dir)
            action["status"] = ActionStatus.EXECUTED.value
            action["timestamps"]["executed_at"] = utc_now()
            action["execution_result"] = {"outcome": result.outcome, **result.details}
            event = "executed"
        except Exception as exc:
            action["status"] = ActionStatus.FAILED.value
            action["timestamps"]["failed_at"] = utc_now()
            # Keep the exception class, not its message, to prevent credential leakage.
            action["execution_result"] = {"error": f"{type(exc).__name__}; execution was blocked or failed safely."}
            event = "failed"
        self.store.save_actions(actions)
        self.store.append_audit({"event": event, "action_id": action_id, "actor": actor, "approval": action.get("approval"),
                                 "execution_result": action.get("execution_result"), "evidence": action.get("evidence"),
                                 "policy_sources": action.get("policy_sources"), "timestamp": utc_now()})
        return action

    def run_dry_run(self, report_path: str | Path = "output/agentic_financial_analysis.json") -> dict[str, Any]:
        self.phase6_report_path = Path(report_path)
        proposals = self.actions.propose_from_phase6(report_path)
        pending_ids = [a["action_id"] for a in self.pending()]
        results = {"status": "dry_run_complete", "created_action_ids": proposals["created_action_ids"],
                   "reused_action_ids": proposals["reused_action_ids"], "rejected_recommendations": proposals["rejected_recommendations"],
                   "pending_approval_ids": pending_ids, "executed_action_ids": [], "dry_run": True,
                   "human_approval_required": True,
                   "message": "Proposals were validated and recorded. No action ran because a human has not approved them."}
        self.store.save_results(results)
        self.store.write_summary(_summary(results, self.store.load_actions()))
        self.store.append_audit({"event": "workflow_dry_run", "created_action_ids": results["created_action_ids"],
                                 "pending_approval_ids": pending_ids, "no_side_effects": True, "timestamp": utc_now()})
        return results


def _summary(result: dict[str, Any], actions: list[dict[str, Any]]) -> str:
    lines = ["# Phase 7 Workflow Dry Run", "", result["message"], "",
             f"Proposals created: {len(result['created_action_ids'])}",
             f"Proposals reused (idempotent): {len(result['reused_action_ids'])}",
             f"Pending human approval: {len(result['pending_approval_ids'])}",
             f"Rejected by validation: {len(result['rejected_recommendations'])}"]
    if not actions:
        lines.extend(["", "The Phase 6 report contained no recommendations, so there were no action proposals to create."])
    lines.extend(["", "## Proposals", ""])
    for action in actions:
        lines.append(f"- `{action['action_id']}` — {action['type']} / {action['priority']} / {action['status']}: {action['description']}")
    lines.extend(["", "No action was executed in dry-run mode.", ""])
    return "\n".join(lines)
