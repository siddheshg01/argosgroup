"""Bounded query routing and synthesis across the Phase 1–5 specialist agents."""
from __future__ import annotations
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any
from .financial_analyst import FinancialAnalystAgent
from .forecast_agent import ForecastAgent
from .models import AgentRequest, AgentResponse
from .policy_rag_agent import PolicyRAGAgent
from .recommendation_agent import RecommendationAgent
from .root_cause_agent import RootCauseAgent

logger = logging.getLogger(__name__)
MAX_AGENT_CALLS = 5


def plan_agents(question: str) -> tuple[list[str], dict[str, str]]:
    """Route via deterministic intent rules; recommendations require all evidence agents."""
    q = question.casefold()
    asks_profit_action = bool(re.search(r"\bprofit(?:s|ability)?\b|\bmargin\b", q) and re.search(
        r"\b(improv\w*|increas\w*|gain\w*|grow\w*|boost\w*|maximi[sz]\w*|rais\w*|recommend\w*|should|how can|how to|what should)\b", q))
    wants_rec = asks_profit_action or any(term in q for term in ("recommend", "should i", "should we", "next step", "what should", "suggest"))
    wants_root = any(term in q for term in ("why", "root cause", "contribut", "declin", "increas", "change", "driver"))
    wants_forecast = any(term in q for term in ("forecast", "future", "next month", "next quarter", "predict", "expected", "outlook"))
    wants_policy = any(term in q for term in ("policy", "procedure", "rule", "compliant", "consistent", "approval", "control"))
    wants_financial = any(term in q for term in ("financial", "revenue", "sales", "order", "quantity", "profit", "risk", "customer", "category", "location", "performance", "trend"))
    if wants_rec:
        selected = ["financial_analyst", "root_cause_agent", "forecast_agent", "policy_rag_agent", "recommendation_agent"]
    else:
        selected = []
        if wants_financial or not (wants_root or wants_forecast or wants_policy):
            selected.append("financial_analyst")
        if wants_root: selected.append("root_cause_agent")
        if wants_forecast: selected.append("forecast_agent")
        if wants_policy: selected.append("policy_rag_agent")
    dependencies = {"recommendation_agent": "financial/root_cause/forecast/policy agents must run first"} if wants_rec else {}
    return selected[:MAX_AGENT_CALLS], dependencies


def _report_paths_exist() -> bool:
    return all(Path(p).is_file() for p in (
        "output/financial_report.json", "output/root_cause_report.json", "output/forecast_report.json",
        "output/llm_financial_analysis.json", "output/rag_report.json"))


def _recent_months_answer(result: dict[str, Any]) -> str:
    """Render a requested recent monthly series from evidence already returned by an agent."""
    question = str(result.get("question", "")).casefold()
    if not re.search(r"\b(revenue|sales)\b", question):
        return ""
    match = re.search(r"\blast\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve)\s+months?\b", question)
    if not match:
        return ""
    month_count = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                   "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12}.get(match.group(1), None)
    month_count = month_count or int(match.group(1))
    if month_count < 1:
        return ""

    monthly = []
    for agent in result.get("agent_results", []):
        if not isinstance(agent, dict):
            continue
        evidence = agent.get("evidence", {})
        trend = evidence.get("trend_analysis", {}) if isinstance(evidence, dict) else {}
        rows = trend.get("monthly", []) if isinstance(trend, dict) else []
        if isinstance(rows, list):
            monthly = [row for row in rows if isinstance(row, dict) and row.get("_month") is not None and isinstance(row.get("revenue"), (int, float))]
            if monthly:
                break
    if not monthly:
        return ""

    selected = monthly[-month_count:]
    total = sum(float(row["revenue"]) for row in selected)
    lines = [f"Revenue for the {len(selected)} most recent observed months ({selected[0]['_month']} to {selected[-1]['_month']}):"]
    lines.extend(f"- {row['_month']}: ₹{float(row['revenue']):,.2f}" for row in selected)
    lines.append(f"Total: ₹{total:,.2f}")
    return "\n".join(lines)


def final_answer_text(result: dict[str, Any]) -> str:
    """Extract an existing answer or render the structured agent findings as prose."""
    for key in ("final_answer", "answer", "response"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            text = value.get("text") or value.get("answer")
            if isinstance(text, str) and text.strip():
                return text.strip()

    monthly_answer = _recent_months_answer(result)
    if monthly_answer:
        return monthly_answer

    analysis = result.get("analysis")
    if isinstance(analysis, dict):
        for key in ("final_answer", "answer", "response"):
            value = analysis.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        headings = {"facts": "Financial findings", "findings": "Investigation findings",
                    "forecasts": "Forecast findings", "policies": "Policy findings",
                    "recommendations": "Recommendations"}
        sections = []
        for key, heading in headings.items():
            items = analysis.get(key)
            if not isinstance(items, list):
                continue
            texts = [item["text"].strip() for item in items
                     if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip()]
            if texts:
                sections.append(f"{heading}:\n" + "\n".join(f"- {text}" for text in texts))
        if sections:
            return "\n\n".join(sections)

    for agent in result.get("agent_results", []):
        if not isinstance(agent, dict):
            continue
        for key in ("final_answer", "answer", "response"):
            value = agent.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Agentic Financial Analysis", "", f"**Question:** {report['question']}",
             f"**Status:** {report['status']}", "", "## Facts", ""]
    categories = ("facts", "findings", "forecasts", "policies", "recommendations")
    for category in categories:
        if category != "facts":
            lines.extend(["## " + category.title(), ""])
        items = report["analysis"][category]
        if not items:
            lines.extend(["_No evidence returned._", ""])
        for item in items:
            text = item.get("text", "")
            refs = ", ".join(item.get("evidence_refs", item.get("source_chunk_ids", [])))
            lines.append(f"- {text}" + (f" _(Evidence: {refs})_" if refs else ""))
        lines.append("")
    lines.extend(["## Sources", ""])
    for source in report.get("sources", []):
        loc = f", page {source['page']}" if source.get("page") else ""
        section = f", section {source['section']}" if source.get("section") else ""
        lines.append(f"- {source.get('source', 'unknown')}{loc}{section}")
    if not report.get("sources"):
        lines.append("_No external policy source was retrieved._")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report.get("limitations", []))
    lines.extend(["", "## Execution trace", "", "| Agent | Status | Duration (s) |", "|---|---|---:|"])
    for entry in report["trace"]:
        lines.append(f"| {entry['agent']} | {entry['status']} | {entry['duration_seconds']:.2f} |")
    lines.append("")
    return "\n".join(lines)


class OrchestratorAgent:
    """Execute each selected agent once, in order, under a hard call limit."""
    def __init__(self, agents: dict[str, Any] | None = None, max_agent_calls: int = MAX_AGENT_CALLS):
        self.agents = agents or {
            "financial_analyst": FinancialAnalystAgent(), "root_cause_agent": RootCauseAgent(),
            "forecast_agent": ForecastAgent(), "policy_rag_agent": PolicyRAGAgent(),
            "recommendation_agent": RecommendationAgent(),
        }
        self.max_agent_calls = min(MAX_AGENT_CALLS, max(1, int(max_agent_calls)))

    def run(self, question: str, output_dir: str | Path = "output") -> dict[str, Any]:
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        correlation_id = uuid.uuid4().hex[:12]
        selected, dependencies = plan_agents(question)
        selected = selected[:self.max_agent_calls]
        responses: list[AgentResponse] = []
        trace: list[dict[str, Any]] = []
        seen: set[str] = set()
        for name in selected:
            if name in seen or len(responses) >= self.max_agent_calls:
                trace.append({"agent": name, "status": "blocked", "duration_seconds": 0.0,
                              "reason": "loop or maximum-agent-call protection"})
                continue
            seen.add(name)
            agent = self.agents.get(name)
            if agent is None:
                response = AgentResponse(name, "unavailable", limitations=[f"Agent not configured: {name}"])
                elapsed = 0.0
            else:
                request = AgentRequest(question.strip(), {"agent_responses": list(responses)}, correlation_id,
                                       {"dependencies": dependencies.get(name, "")})
                start = time.perf_counter()
                try:
                    response = agent.run(request)
                    if not isinstance(response, AgentResponse):
                        raise TypeError(f"Agent {name} must return AgentResponse")
                except Exception as exc:
                    logger.exception("Agent %s failed for request %s", name, correlation_id)
                    response = AgentResponse(name, "error", limitations=[f"Agent failed: {type(exc).__name__}: {exc}"])
                elapsed = time.perf_counter() - start
            responses.append(response)
            trace.append({"agent": name, "status": response.status, "duration_seconds": round(elapsed, 4),
                          "dependencies": dependencies.get(name, ""), "correlation_id": correlation_id})
            logger.info("agent=%s status=%s elapsed=%.3f request=%s", name, response.status, elapsed, correlation_id)

        groups: dict[str, list[dict[str, Any]]] = {k: [] for k in ("facts", "findings", "forecasts", "policies", "recommendations")}
        sources: list[dict[str, Any]] = []
        limitations: list[str] = []
        for response in responses:
            limitations.extend(response.limitations)
            for source in response.sources:
                if source not in sources: sources.append(source)
            category = {"financial_analyst": "facts", "root_cause_agent": "findings", "forecast_agent": "forecasts",
                        "policy_rag_agent": "policies", "recommendation_agent": "recommendations"}.get(response.agent)
            if category: groups[category].extend(response.findings)
        output = {"request_id": correlation_id, "question": question.strip(), "status": "complete" if all(r.status != "error" for r in responses) else "partial",
                  "selected_agents": selected, "agent_results": [r.to_dict() for r in responses],
                  "analysis": groups, "sources": sources, "limitations": list(dict.fromkeys(limitations)),
                  "trace": trace, "limits": {"max_agent_calls": self.max_agent_calls, "actual_agent_calls": len(responses), "each_agent_once": True},
                  "phase_1_to_5_integrated": _report_paths_exist(),
                  "capabilities": {"gemini": any(r.agent == "recommendation_agent" and r.status == "success" and r.evidence.get("generator") != "deterministic_fallback" for r in responses) or any(r.agent == "policy_rag_agent" and r.status == "success" and r.evidence.get("status") == "answered" for r in responses),
                                   "rag": any(r.agent == "policy_rag_agent" and r.status == "success" for r in responses),
                                   "recommendations": any(r.agent == "recommendation_agent" and r.status == "success" for r in responses)}}
        output["final_answer"] = final_answer_text(output)
        out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "agentic_financial_analysis.json").write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        (out_dir / "agentic_financial_analysis.md").write_text(_render_markdown(output), encoding="utf-8")
        return output


def run_agent_query(question: str, output_dir: str | Path = "output") -> dict[str, Any]:
    """Programmatic public interface for asking a bounded multi-agent question."""
    return OrchestratorAgent().run(question, output_dir)
