"""Phase 3 report adapter for supported forecasts only."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .models import AgentRequest, AgentResponse


class ForecastAgent:
    name = "forecast_agent"
    def __init__(self, report_path: str | Path = "output/forecast_report.json"):
        self.report_path = Path(report_path)

    def run(self, request: AgentRequest) -> AgentResponse:
        if not self.report_path.is_file():
            return AgentResponse(self.name, "unavailable", limitations=[f"Phase 3 report not found: {self.report_path}"])
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        words = set(request.question.lower().replace("/", " ").split())
        metrics = {m for m in ("revenue", "quantity", "orders") if m in words}
        forecasts = report.get("forecasts", [])
        if metrics:
            forecasts = [item for item in forecasts if item.get("metric") in metrics]
        forecasts = [item for item in forecasts if item.get("status") == "forecasted"]
        findings = [{"classification": "forecast", "text": f"Phase 3 forecast is available for {item.get('metric')} with trend {item.get('trend', {}).get('direction', 'not stated')}.",
                     "evidence_refs": ["phase3"], "metric": item.get("metric"),
                     "selected_model": item.get("selected_model"), "validation_metrics": item.get("validation_metrics"),
                     "forecasts_by_horizon": item.get("forecasts_by_horizon"), "trend": item.get("trend"),
                     "significant_changes": item.get("significant_changes"), "assumptions": item.get("assumptions", [])}
                    for item in forecasts]
        limitations = [] if findings else ["No supported forecast matching the requested metrics was present in the Phase 3 report."]
        return AgentResponse(self.name, "success", findings or [{"classification": "fact", "text": "No matching forecast is available in Phase 3.", "evidence_refs": ["phase3"]}],
                             {"metrics_forecasted": report.get("metrics_forecasted", []), "forecasts": forecasts},
                             [{"source": str(self.report_path), "phase": 3}], limitations)
