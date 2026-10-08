"""CLI for proposing, approving, rejecting, executing, and dry-running actions."""
from __future__ import annotations
import argparse
import json
from .actions import ActionType
from .approval import ActionStatus
from .workflow import WorkflowEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 7 approval-gated workflow CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    dry = sub.add_parser("dry-run", help="Create proposals from Phase 6 without executing actions")
    dry.add_argument("--phase6-report", default="output/agentic_financial_analysis.json")
    propose = sub.add_parser("propose", help="Create pending action proposals from Phase 6 recommendations")
    propose.add_argument("--phase6-report", default="output/agentic_financial_analysis.json")
    propose.add_argument("--type", choices=[x.value for x in ActionType])
    sub.add_parser("pending", help="List actions awaiting human approval")
    for decision in ("approve", "reject"):
        cmd = sub.add_parser(decision, help=f"{decision.title()} one pending action")
        cmd.add_argument("action_id")
        cmd.add_argument("--approver", required=True, help="Human operator name or ID")
        cmd.add_argument("--reason", default="")
    execute = sub.add_parser("execute", help="Execute only an approved action")
    execute.add_argument("action_id")
    execute.add_argument("--dry-run", action="store_true", help="Simulate execution without side effects")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    engine = WorkflowEngine()
    if args.command == "dry-run":
        result = engine.run_dry_run(args.phase6_report)
        print(f"Dry-run workflow: {result['status']}")
        print(f"Proposals created: {len(result['created_action_ids'])}; pending approval: {len(result['pending_approval_ids'])}")
        print("No actions were executed.")
    elif args.command == "propose":
        result = engine.actions.propose_from_phase6(args.phase6_report, args.type)
        print(f"Created {len(result['created_action_ids'])} pending proposal(s); reused {len(result['reused_action_ids'])}.")
        if result["rejected_recommendations"]:
            print(f"Validation rejected {len(result['rejected_recommendations'])} recommendation(s).")
    elif args.command == "pending":
        print(json.dumps(engine.pending(), indent=2, ensure_ascii=False))
    elif args.command == "approve":
        action = engine.approve(args.action_id, args.approver, args.reason)
        print(f"Approved {action['action_id']} by {action['approval']['approver']}.")
    elif args.command == "reject":
        action = engine.reject(args.action_id, args.approver, args.reason)
        print(f"Rejected {action['action_id']} by {action['approval']['approver']}.")
    elif args.command == "execute":
        result = engine.execute(args.action_id, dry_run=args.dry_run)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if result["status"] == "blocked":
            raise SystemExit(2)
    print("Reports: output/action_proposals.json, output/workflow_results.json, output/workflow_audit.json, output/workflow_summary.md")


if __name__ == "__main__":
    main()
