"""Local JSON persistence for proposals, workflow results, and approval audit events."""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


class WorkflowStore:
    def __init__(self, output_dir: str | Path = "output"):
        self.output_dir = Path(output_dir)
        self.proposals_path = self.output_dir / "action_proposals.json"
        self.results_path = self.output_dir / "workflow_results.json"
        self.audit_path = self.output_dir / "workflow_audit.json"
        self.summary_path = self.output_dir / "workflow_summary.md"

    def load_actions(self) -> list[dict[str, Any]]:
        if not self.proposals_path.exists(): return []
        content = json.loads(self.proposals_path.read_text(encoding="utf-8"))
        return content.get("actions", []) if isinstance(content, dict) else content

    def save_actions(self, actions: list[dict[str, Any]]) -> None:
        write_json_atomic(self.proposals_path, {"actions": actions})

    def append_audit(self, event: dict[str, Any]) -> None:
        events = json.loads(self.audit_path.read_text(encoding="utf-8")) if self.audit_path.exists() else []
        events.append(event)
        write_json_atomic(self.audit_path, events)

    def save_results(self, results: dict[str, Any]) -> None:
        write_json_atomic(self.results_path, results)

    def write_summary(self, text: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(text, encoding="utf-8")
