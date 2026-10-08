"""Phase 2 report adapter that labels contributors as associations, not causes."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .models import AgentRequest, AgentResponse


class RootCauseAgent:
    name = "root_cause_agent"
    def __init__(self, report_path: str | Path = "output/root_cause_report.json"):
        self.report_path = Path(report_path)

    def run(self, request: AgentRequest) -> AgentResponse:
        if not self.report_path.is_file():
            return AgentResponse(self.name, "unavailable", limitations=[f"Phase 2 report not found: {self.report_path}"])
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        words = set(request.question.lower().replace("/", " ").split())
        metrics = {m for m in ("revenue", "quantity", "orders") if m in words}
        if any(word in words for word in ("profit", "profits", "profitability", "margin")):
            metrics.update(("revenue", "quantity", "orders"))
        changes = report.get("significant_changes", [])
        if metrics:
            changes = [item for item in changes if item.get("metric") in metrics]
        # Show the latest observed period instead of ranking noisy percentage spikes
        # from tiny historical baselines ahead of the current business picture.
        periods = [str(item.get("period")) for item in changes if item.get("period")]
        latest_period = max(periods) if periods else None
        if latest_period:
            changes = [item for item in changes if str(item.get("period")) == latest_period]
        changes = sorted(changes, key=lambda item: ("revenue", "quantity", "orders").index(item.get("metric")) if item.get("metric") in ("revenue", "quantity", "orders") else 99)[:3]
        findings = []
        for item in changes:
            raw_contributors = item.get("contributors", [])
            if isinstance(raw_contributors, dict):
                contributors = {dimension: values[:5] if isinstance(values, list) else values
                                for dimension, values in raw_contributors.items()}
            elif isinstance(raw_contributors, list):
                contributors = raw_contributors[:5]
            else:
                contributors = []
            pct = item.get("percentage_change")
            direction = "increased" if isinstance(pct, (int, float)) and pct >= 0 else "decreased"
            contributors_by_dimension = raw_contributors if isinstance(raw_contributors, dict) else {}
            category_contributors = contributors_by_dimension.get("category", []) if isinstance(contributors_by_dimension, dict) else []
            direction_sign = 1 if isinstance(pct, (int, float)) and pct >= 0 else -1
            same_direction = [entry for entry in category_contributors if isinstance(entry, dict) and float(entry.get("absolute_change") or 0) * direction_sign > 0]
            offsets = [entry for entry in category_contributors if isinstance(entry, dict) and float(entry.get("absolute_change") or 0) * direction_sign < 0]
            top_category = max(same_direction, key=lambda entry: abs(float(entry.get("absolute_change") or 0)), default=None)
            offset_category = max(offsets, key=lambda entry: abs(float(entry.get("absolute_change") or 0)), default=None)
            def movement(entry):
                before, after = float(entry.get("previous_value") or 0), float(entry.get("current_value") or 0)
                if item.get("metric") == "revenue":
                    return f"{entry.get('value')} (₹{before:,.0f} to ₹{after:,.0f})"
                return f"{entry.get('value')} ({before:,.0f} to {after:,.0f} units)"
            category_note = f" The largest associated category movement was {movement(top_category)}." if top_category else ""
            if offset_category:
                category_note += f" {movement(offset_category)} moved the other way, offsetting part of the change."
            change_text = f"{item.get('metric', 'Metric').title()} {direction} {abs(pct):.1f}% in {item.get('period')} versus the previous observed month." if isinstance(pct, (int, float)) else f"{item.get('metric', 'Metric').title()} changed in {item.get('period')} versus the previous observed month."
            findings.append({"classification": "finding", "text": change_text + category_note + " These are associations, not proven causes.",
                             "evidence_refs": ["phase2"], "period": item.get("period"), "contributors": contributors,
                             "evidence": item.get("evidence", [])})
        limitations = [] if findings else ["No significant Phase 2 changes matched the requested metrics."]
        if not findings:
            findings = [{"classification": "fact", "text": "No significant matching change was present in the Phase 2 report.", "evidence_refs": ["phase2"]}]
        return AgentResponse(self.name, "success", findings,
                             {"metrics_investigated": report.get("metrics_investigated", []), "significant_change_count": report.get("significant_change_count"), "significant_changes": changes},
                             [{"source": str(self.report_path), "phase": 2}], limitations)
