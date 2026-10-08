"""FastAPI adapter exposing saved Phase 1–7 intelligence and controlled workflows."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"
REPORTS = {
    "financial": "financial_report.json", "root_cause": "root_cause_report.json",
    "forecast": "forecast_report.json", "analyst": "llm_financial_analysis.json",
    "rag": "rag_report.json", "agents": "agentic_financial_analysis.json",
    "proposals": "action_proposals.json", "workflow": "workflow_results.json",
    "audit": "workflow_audit.json",
    "financial_csv": "financial_summary.csv", "root_cause_csv": "root_cause_summary.csv",
    "forecast_csv": "forecast_summary.csv", "analyst_md": "llm_financial_analysis.md",
    "rag_md": "rag_summary.md", "agents_md": "agentic_financial_analysis.md",
    "workflow_md": "workflow_summary.md",
}

app = FastAPI(title="FINOP API", version="1.0.0", description="API adapter for the existing FINOP Phases 1–7.")
logger = logging.getLogger(__name__)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("FINOP_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
                   allow_credentials=True, allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])


def authorized(authorization: str | None = Header(default=None)) -> str:
    """Authentication-ready bearer-token gate; set FINOP_API_TOKEN outside local development."""
    expected = os.getenv("FINOP_API_TOKEN")
    if expected and authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Authentication required")
    return "authenticated-user" if expected else "local-development-user"


def read_report(name: str, default: Any = None) -> Any:
    path = OUTPUT / REPORTS[name]
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Report '{name}' could not be read ({type(exc).__name__}).") from None


ANALYST_SECTION_LABELS = {
    "executive_summary": "Executive summary", "performance_analysis": "Performance",
    "root_cause_analysis": "Root-cause findings", "forecast_analysis": "Forecast",
    "trends": "Trends", "risks": "Risks", "opportunities": "Opportunities",
    "attention_items": "Management attention", "recommendations": "Recommended next steps",
    "data_limitations": "Data limitations",
}


def format_analyst_answer(analysis: dict[str, Any]) -> str:
    """Render Phase 4's grounded structured items into a readable answer string."""
    sections = []
    for key, heading in ANALYST_SECTION_LABELS.items():
        items = analysis.get(key, [])
        texts = [str(item["text"]).strip() for item in items
                 if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip()]
        if texts:
            sections.append(f"{heading}\n" + "\n".join(f"- {text}" for text in texts))
    return "\n\n".join(sections)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class ActionTypeRequest(BaseModel):
    type: Literal["email_notification", "financial_alert", "management_report", "ticket_task"] | None = None


class DecisionRequest(BaseModel):
    actor: str = Field(min_length=1, max_length=120)
    reason: str = Field(default="", max_length=1000)


class ExecuteRequest(BaseModel):
    actor: str = Field(min_length=1, max_length=120)
    confirm: bool
    dry_run: bool = False


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "phases": [1, 2, 3, 4, 5, 6, 7], "available_reports": [k for k in REPORTS if (OUTPUT / REPORTS[k]).is_file()]}


@app.get("/api/dashboard")
def dashboard(_: str = Depends(authorized)) -> dict[str, Any]:
    financial = read_report("financial", {})
    root_cause = read_report("root_cause", {})
    forecast = read_report("forecast", {})
    analyst = read_report("analyst", {})
    agents = read_report("agents", {})
    proposals = read_report("proposals", {"actions": []})
    monthly = financial.get("trend_analysis", {}).get("monthly", [])
    risk = financial.get("risk_analysis", {})
    return {"financial_summary": financial.get("financial_summary", {}), "kpis": financial.get("kpis", {}),
            "risk": risk, "monthly_trends": monthly, "latest_root_cause": root_cause.get("significant_changes", [])[:5],
            "forecast_summary": forecast.get("forecasts", []), "attention_items": analyst.get("attention_items", []),
            "recommendations": agents.get("analysis", {}).get("recommendations", []),
            "recent_alerts": [a for a in proposals.get("actions", []) if a.get("type") == "financial_alert" or a.get("priority") in {"HIGH", "CRITICAL"}][:8],
            "data_status": {"phase1": bool(financial), "phase2": bool(root_cause), "phase3": bool(forecast)}}


@app.get("/api/financial-intelligence")
def financial_intelligence(_: str = Depends(authorized)) -> dict[str, Any]:
    return read_report("financial", {})


@app.get("/api/root-cause")
def root_cause(metric: str | None = Query(default=None, pattern="^(revenue|quantity|orders)$"), _: str = Depends(authorized)) -> dict[str, Any]:
    report = read_report("root_cause", {})
    if metric:
        report = {**report, "significant_changes": [x for x in report.get("significant_changes", []) if x.get("metric") == metric]}
    return report


@app.get("/api/forecast")
def forecasting(metric: str | None = Query(default=None, pattern="^(revenue|quantity|orders)$"), _: str = Depends(authorized)) -> dict[str, Any]:
    report = read_report("forecast", {})
    if metric:
        report = {**report, "forecasts": [x for x in report.get("forecasts", []) if x.get("metric") == metric]}
    return report


@app.post("/api/analyst/ask")
def analyst_ask(request: QuestionRequest, _: str = Depends(authorized)) -> dict[str, Any]:
    """Use the existing Phase 4 context, prompt, Gemini caller, and evidence validator."""
    try:
        financial = read_report("financial", {})
        financial_metadata = financial.get("metadata", {})
        unavailable = financial_metadata.get("unavailable_fields", [])
        kpis = financial.get("kpis", {})
        asks_profit = re.search(r"\bprofit(?:s|ability)?\b|\bmargin\b", request.question, re.IGNORECASE)
        asks_profit_action = asks_profit and re.search(
            r"\b(improv\w*|increas\w*|gain\w*|grow\w*|boost\w*|maximi[sz]\w*|rais\w*|recommend\w*|should|how can|how to|what should)\b",
            request.question, re.IGNORECASE,
        )
        if asks_profit_action:
            # Route profitability-improvement requests through the complete evidence
            # workflow even when a client still posts to the direct analyst endpoint.
            from .agents.orchestrator import run_agent_query
            return run_agent_query(request.question, OUTPUT)
        profit_unavailable = (
            "profit" in unavailable or "cost" in unavailable
            or str(kpis.get("profit_status", "")).startswith("not_available")
            or str(kpis.get("cost_status", "")).startswith("not_available")
        )
        if asks_profit and profit_unavailable:
            revenue = kpis.get("total_revenue", financial.get("financial_summary", {}).get("total_revenue"))
            revenue_context = f" Reported revenue is ₹{float(revenue):,.2f}; revenue is not the same as profit." if isinstance(revenue, (int, float)) else " Revenue is not the same as profit."
            answer = (
                "I can’t calculate profit from the current dataset. It includes sales and shipping charges, "
                "but has no product cost/COGS or profit field, so subtracting shipping alone would not give "
                "a valid profit figure. Add product costs (and confirm whether shipping, tax, and returns "
                "should be included) to calculate it."
                + revenue_context
            )
            return {
                "request_id": uuid4().hex, "question": request.question, "status": "answered", "answer": answer,
                "analysis": {
                    "executive_summary": [{"text": answer, "classification": "fact", "evidence_ids": ["phase1_financial_summary"]}],
                    "data_limitations": [{"text": "Product cost/COGS and a profit field are not available in the current source dataset.", "classification": "limitation", "evidence_ids": ["phase1_financial_summary"]}],
                },
                "sources": [{"source": "financial_report.json"}], "selected_agents": ["financial_analyst"],
                "limitations": ["Profit cannot be calculated without product cost/COGS data."],
            }
        profit_assumption = financial_metadata.get("feature_formulas", {}).get("profit", {}).get("assumption", "")
        synthetic_profit = "SYNTHETIC ESTIMATE" in str(profit_assumption)
        if asks_profit and synthetic_profit:
            profit = kpis.get("total_profit")
            revenue = kpis.get("total_revenue")
            margin = kpis.get("profit_margin_pct")
            if all(isinstance(value, (int, float)) for value in (profit, revenue, margin)):
                answer = (
                    f"For the 5,000-row sample, estimated profit is ₹{profit:,.2f} "
                    f"({margin:.2f}% estimated margin) on ₹{revenue:,.2f} reported revenue. "
                    "These figures use synthetic COGS assumptions, not actual supplier or accounting costs; "
                    "treat them as illustrative only and replace the assumed costs before making business decisions."
                )
                return {
                    "request_id": uuid4().hex, "question": request.question, "status": "answered", "answer": answer,
                    "analysis": {
                        "executive_summary": [{"text": answer, "classification": "fact", "evidence_ids": ["phase1_financial_summary"]}],
                        "data_limitations": [{"text": str(profit_assumption), "classification": "limitation", "evidence_ids": ["phase1_financial_summary"]}],
                    },
                    "sources": [{"source": "financial_report.json"}], "selected_agents": ["financial_analyst"],
                    "limitations": ["Profit and margin are synthetic estimates, not actual financial results."],
                }
        from .llm_analyst import call_gemini, load_local_env, parse_response, validate_analysis
        from .prompt_builder import build_analysis_context, build_prompt
        load_local_env(ROOT / ".env")
        key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
        if not key:
            from .agents.orchestrator import run_agent_query
            fallback = run_agent_query(request.question, OUTPUT)
            answer = fallback.get("final_answer") or ""
            fallback["answer"] = answer or "No report-grounded findings were available for this question."
            fallback["status"] = "answered" if answer and fallback.get("status") == "complete" else (
                "partial" if answer else "insufficient_evidence"
            )
            limitations = fallback.get("limitations")
            if not isinstance(limitations, list):
                limitations = []
            limitations.append("Gemini is not configured; this answer uses deterministic report-based analysis.")
            fallback["limitations"] = list(dict.fromkeys(limitations))
            return fallback
        sources = [read_report(key) for key in ("financial", "root_cause", "forecast")]
        context = build_analysis_context(*sources)
        prompt = build_prompt(context) + "\n\nUSER QUESTION (answer within the same evidence-grounded sections):\n" + request.question
        # Match Phase 4's environment precedence so GEMINI_MODEL overrides the
        # existing LLM_MODEL setting, with the same default as a last resort.
        model = os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL") or "gemini-3.1-flash-lite"
        result = call_gemini(prompt, key, model, float(os.getenv("LLM_TIMEOUT_SECONDS", "45")))
        # Phase 4 already provides the canonical parser for Gemini response envelopes;
        # it handles both provider envelopes and plain JSON before evidence validation.
        parsed = validate_analysis(parse_response(result), context)
        answer = format_analyst_answer(parsed)
        if not answer:
            logger.warning("AI analyst returned a valid but empty grounded analysis for request %s", request.question[:120])
            return {"request_id": uuid4().hex, "question": request.question, "status": "insufficient_evidence",
                    "answer": "The analyst did not return any evidence-grounded findings for this question.",
                    "analysis": parsed, "sources": list(context["evidence_ids"]),
                    "limitations": ["Gemini returned no supported analysis items for this question."]}
        return {"request_id": uuid4().hex, "question": request.question, "status": "answered", "answer": answer, "analysis": parsed,
                "sources": list(context["evidence_ids"]), "limitations": ["Answers are restricted to Phase 1–3 report evidence."]}
    except HTTPException:
        raise
    except Exception as exc:
        request_id = uuid4().hex
        logger.exception("AI analyst request %s failed (exception_type=%s)", request_id, type(exc).__name__)
        raise HTTPException(status_code=502, detail=f"Financial analyst unavailable ({type(exc).__name__}).") from None


@app.post("/api/policy/ask")
def policy_ask(request: QuestionRequest, _: str = Depends(authorized)) -> dict[str, Any]:
    try:
        from .rag.rag_pipeline import RAGPipeline
        return RAGPipeline(knowledge_dir=ROOT / "data/knowledge_base", vector_dir=ROOT / "data/processed/policy_chroma", output_dir=OUTPUT).run(query=request.question)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Policy retrieval unavailable ({type(exc).__name__}).") from None


@app.post("/api/agents/query")
def agents_query(request: QuestionRequest, _: str = Depends(authorized)) -> dict[str, Any]:
    try:
        from .agents.orchestrator import run_agent_query
        return run_agent_query(request.question, OUTPUT)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Agent query unavailable ({type(exc).__name__}).") from None


@app.get("/api/agents/latest")
def agents_latest(_: str = Depends(authorized)) -> dict[str, Any]:
    return read_report("agents", {})


@app.get("/api/analyst/report")
def analyst_report(_: str = Depends(authorized)) -> dict[str, Any]:
    return read_report("analyst", {})


@app.get("/api/policy/report")
def policy_report(_: str = Depends(authorized)) -> dict[str, Any]:
    return read_report("rag", {})


@app.get("/api/alerts")
def alerts(_: str = Depends(authorized)) -> dict[str, Any]:
    financial = read_report("financial", {})
    workflow = read_report("proposals", {"actions": []}).get("actions", [])
    audit_events = read_report("audit", [])
    risk = financial.get("risk_analysis", {})
    items = [{"id": f"risk-{i}", "type": "risk", "status": risk.get("risk_level", "unknown"),
              "description": text, "source": "Phase 1 risk report"}
             for i, text in enumerate(risk.get("risk_factors", []))]
    items.extend({"id": action.get("action_id"), "type": "workflow", "status": action.get("status"),
                  "description": action.get("description"), "priority": action.get("priority"), "source": "Phase 7 action workflow"}
                 for action in workflow if action.get("status") in {"PENDING", "FAILED"})
    items.extend({"id": f"audit-{i}", "type": "workflow", "status": event.get("event", "unknown"),
                  "description": event.get("reason") or event.get("event"), "source": "Phase 7 audit", "timestamp": event.get("timestamp")}
                 for i, event in enumerate(audit_events[-20:]) if event.get("event") in {"failed", "execution_blocked", "proposal_validation_failed"})
    return {"alerts": items, "risk_level": risk.get("risk_level"), "risk_score": risk.get("risk_score")}


@app.get("/api/recommendations")
def recommendations(_: str = Depends(authorized)) -> dict[str, Any]:
    report = read_report("agents", {})
    return {"recommendations": report.get("analysis", {}).get("recommendations", []), "status": report.get("status", "unavailable")}


@app.get("/api/actions")
def actions(status: str | None = None, _: str = Depends(authorized)) -> dict[str, Any]:
    result = read_report("proposals", {"actions": []})
    items = result.get("actions", [])
    return {"actions": [a for a in items if status is None or a.get("status") == status]}


@app.post("/api/actions/propose")
def propose_actions(request: ActionTypeRequest | None = None, _: str = Depends(authorized)) -> dict[str, Any]:
    from .workflow import WorkflowEngine
    try:
        engine = WorkflowEngine(OUTPUT, OUTPUT / REPORTS["agents"])
        return engine.actions.propose_from_phase6(OUTPUT / REPORTS["agents"], request.type if request else None)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Could not create proposals ({type(exc).__name__}).") from None


@app.get("/api/actions/pending")
def pending_actions(_: str = Depends(authorized)) -> dict[str, Any]:
    return {"actions": read_report("proposals", {"actions": []}).get("actions", [])}


@app.post("/api/actions/{action_id}/approve")
def approve_action(action_id: str, request: DecisionRequest, _: str = Depends(authorized)) -> dict[str, Any]:
    from .workflow import WorkflowEngine
    try:
        return WorkflowEngine(OUTPUT).approve(action_id, request.actor, request.reason)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


@app.post("/api/actions/{action_id}/reject")
def reject_action(action_id: str, request: DecisionRequest, _: str = Depends(authorized)) -> dict[str, Any]:
    from .workflow import WorkflowEngine
    try:
        return WorkflowEngine(OUTPUT).reject(action_id, request.actor, request.reason)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


@app.post("/api/actions/{action_id}/execute")
def execute_action(action_id: str, request: ExecuteRequest, _: str = Depends(authorized)) -> dict[str, Any]:
    if not request.confirm:
        raise HTTPException(status_code=400, detail="Explicit execution confirmation is required.")
    from .workflow import WorkflowEngine
    try:
        return WorkflowEngine(OUTPUT, OUTPUT / REPORTS["agents"]).execute(action_id, dry_run=request.dry_run, actor=request.actor)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Action not found.") from None


@app.get("/api/audit")
def audit(_: str = Depends(authorized)) -> dict[str, Any]:
    return {"events": read_report("audit", [])}


@app.get("/api/reports")
def report_list(_: str = Depends(authorized)) -> dict[str, Any]:
    labels = {"financial": "Financial Intelligence", "root_cause": "Root Cause Investigation", "forecast": "Forecasting",
              "analyst": "AI Financial Analysis", "rag": "Policy / RAG", "agents": "Multi-agent Analysis",
              "proposals": "Action Proposals", "workflow": "Workflow Results", "audit": "Workflow Audit",
              "financial_csv": "Financial Summary CSV", "root_cause_csv": "Root Cause Summary CSV",
              "forecast_csv": "Forecast Summary CSV", "analyst_md": "AI Analysis Markdown",
              "rag_md": "Policy Analysis Markdown", "agents_md": "Agentic Analysis Markdown", "workflow_md": "Workflow Summary Markdown"}
    return {"reports": [{"id": key, "label": labels[key], "filename": REPORTS[key], "available": (OUTPUT / REPORTS[key]).is_file()}
                         for key in REPORTS]}


@app.get("/api/reports/{report_id}")
def download_report(report_id: str, _: str = Depends(authorized)):
    filename = REPORTS.get(report_id)
    if not filename:
        raise HTTPException(status_code=404, detail="Unknown report.")
    path = OUTPUT / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Report is not available yet.")
    media = "application/json" if filename.endswith(".json") else "text/csv" if filename.endswith(".csv") else "text/markdown"
    return FileResponse(path, filename=filename, media_type=media)


# Serve the built React app from the same origin in production. During development,
# Vite serves the frontend and proxies /api to FastAPI instead.
WEB_DIST = ROOT / "web" / "dist"
if (WEB_DIST / "index.html").is_file():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="finop-web")
