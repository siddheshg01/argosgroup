"""Phase 1 report adapter for the multi-agent system."""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any
from .models import AgentRequest, AgentResponse


class FinancialAnalystAgent:
    name = "financial_analyst"
    def __init__(self, report_path: str | Path = "output/financial_report.json"):
        self.report_path = Path(report_path)

    def run(self, request: AgentRequest) -> AgentResponse:
        if not self.report_path.is_file():
            return AgentResponse(self.name, "unavailable", limitations=[f"Phase 1 report not found: {self.report_path}"])
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        evidence = {"financial_summary": report.get("financial_summary", {}),
                    "kpis": report.get("kpis", {}), "data_quality": report.get("data_quality", {}),
                    "metadata": report.get("metadata", {}),
                    "trend_analysis": report.get("trend_analysis", {}),
                    "location_analysis": report.get("location_analysis", [])[:5],
                    "regional_analysis": report.get("regional_analysis", [])[:5],
                    "product_analysis": report.get("product_analysis", [])[:5],
                    "category_analysis": report.get("category_analysis", [])[:5],
                    "concentration_analysis": report.get("concentration_analysis", {}),
                    "risk_analysis": report.get("risk_analysis", {})}
        limitations = []
        unavailable = report.get("metadata", {}).get("unavailable_fields", [])
        if "profit" in unavailable or "cost" in unavailable:
            limitations.append("Profit and cost are unavailable in the source dataset.")
        profit_assumption = report.get("metadata", {}).get("feature_formulas", {}).get("profit", {}).get("assumption", "")
        if "SYNTHETIC ESTIMATE" in profit_assumption:
            limitations.append("Profit and margin are illustrative estimates from synthetic COGS assumptions, not actual financial results.")
        kpis = report.get("kpis", {})
        question = request.question.casefold()
        findings = []
        if re.search(r"\bprofit(?:s|ability)?\b|\bmargin\b", question) and "total_profit" in kpis:
            estimated = " Estimated using synthetic COGS; this is illustrative, not actual accounting profit." if "SYNTHETIC ESTIMATE" in profit_assumption else ""
            findings.append({"classification": "fact", "text": f"Across the {report.get('metadata', {}).get('records_analyzed', 0):,}-row sample, reported profit is ₹{kpis['total_profit']:,.2f} with a {kpis.get('profit_margin_pct', 0):.2f}% margin on ₹{kpis.get('total_revenue', 0):,.2f} revenue.{estimated}", "evidence_refs": ["phase1"]})
            categories = [row for row in report.get("category_analysis", []) if isinstance(row, dict) and isinstance(row.get("profit_margin_pct"), (int, float))]
            if categories:
                top = max(categories, key=lambda row: row["profit_margin_pct"])
                findings.append({"classification": "fact", "text": f"{top.get('Category', 'One category')} has the highest recorded sample margin at {top['profit_margin_pct']:.2f}%. This ranking also relies on the synthetic COGS assumption.", "evidence_refs": ["phase1"]})
        else:
            summary = report.get("financial_summary", {})
            revenue = kpis.get("total_revenue", summary.get("total_revenue", summary.get("revenue")))
            orders = kpis.get("total_orders", summary.get("total_orders", summary.get("orders")))
            facts = []
            if isinstance(revenue, (int, float)):
                facts.append(f"revenue is ₹{revenue:,.2f}")
            profit = kpis.get("total_profit")
            if isinstance(profit, (int, float)) and "profit" not in unavailable:
                profit_label = "estimated profit" if "SYNTHETIC ESTIMATE" in profit_assumption else "profit"
                facts.append(f"{profit_label} is ₹{profit:,.2f}")
            if isinstance(orders, (int, float)):
                facts.append(f"orders total {orders:,.0f}")
            if facts:
                text = f"For the {report.get('metadata', {}).get('records_analyzed', 0):,}-row sample, " + ", ".join(facts) + "."
            else:
                text = "The financial report does not contain available revenue, profit, or order totals for this question."
            findings.append({"classification": "fact", "text": text, "evidence_refs": ["phase1"]})
        return AgentResponse(self.name, "success", findings,
                             evidence, [{"source": str(self.report_path), "phase": 1}], limitations)
