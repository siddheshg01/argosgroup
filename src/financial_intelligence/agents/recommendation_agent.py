"""Evidence-limited Gemini recommendation agent; it never executes actions."""
from __future__ import annotations
import json
import math
import os
import re
import urllib.error
import urllib.request
from typing import Any
from ..llm_analyst import load_local_env
from .models import AgentRequest, AgentResponse

NUMBER_PATTERN = r"(?<!\w)[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?!\w)"


def build_recommendation_context(responses: list[AgentResponse]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate financial, investigation, forecast and policy evidence for synthesis."""
    sections: dict[str, Any] = {}
    source_catalog: dict[str, Any] = {}
    for response in responses:
        if response.agent == "financial_analyst":
            source_catalog["phase1"] = response.evidence
            sections["financial_facts"] = {"findings": response.findings, "evidence": response.evidence, "limitations": response.limitations}
        elif response.agent == "root_cause_agent":
            source_catalog["phase2"] = response.evidence
            sections["root_cause_findings"] = {"findings": response.findings, "evidence": response.evidence, "limitations": response.limitations}
        elif response.agent == "forecast_agent":
            source_catalog["phase3"] = response.evidence
            sections["forecasts"] = {"findings": response.findings, "evidence": response.evidence, "limitations": response.limitations}
        elif response.agent == "policy_rag_agent":
            sections["policy_evidence"] = {"findings": response.findings, "sources": response.sources, "limitations": response.limitations}
            for source in response.sources:
                chunk_id = source.get("chunk_id")
                if chunk_id:
                    source_catalog[f"policy:{chunk_id}"] = next((doc for doc in response.evidence.get("retrieved_documents", []) if doc.get("id") == chunk_id), source)
        else:
            sections[response.agent] = {"findings": response.findings, "limitations": response.limitations}
    return sections, source_catalog


def _call_gemini(prompt: str, allowed_refs: list[str]) -> dict[str, Any]:
    load_local_env()
    key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    primary = os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
    models = list(dict.fromkeys([primary, *[x.strip() for x in os.getenv("GEMINI_FALLBACK_MODELS", os.getenv("LLM_FALLBACK_MODELS", "gemini-3.5-flash-lite")).split(",") if x.strip()]]))
    schema = {"type": "OBJECT", "properties": {
        "recommendations": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "text": {"type": "STRING"}, "evidence_refs": {"type": "ARRAY", "items": {"type": "STRING", "enum": allowed_refs}}},
            "required": ["text", "evidence_refs"]}},
        "limitations": {"type": "ARRAY", "items": {"type": "STRING"}},
    }, "required": ["recommendations", "limitations"]}
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
    last_error: Exception | None = None
    for model in models:
        request_body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema}}
        req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            data=json.dumps(request_body).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": key}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                envelope = json.loads(response.read().decode("utf-8"))
            text = "".join(part.get("text", "") for part in envelope["candidates"][0]["content"]["parts"])
            return json.loads(text)
        except urllib.error.HTTPError as exc:
            try: message = json.loads(exc.read().decode("utf-8", errors="replace")).get("error", {}).get("message", "")
            except (ValueError, AttributeError): message = ""
            last_error = RuntimeError(f"Gemini recommendation API returned HTTP {exc.code}: {message}".rstrip())
            if exc.code not in (404, 429, 500, 503): break
        except urllib.error.URLError as exc:
            last_error = RuntimeError(f"Gemini recommendation network error: {type(exc.reason).__name__}")
            break
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            last_error = ValueError("Gemini returned an invalid recommendation response")
            break
    raise last_error or RuntimeError("Gemini recommendation generation failed")


def validate_recommendations(payload: dict[str, Any], source_catalog: dict[str, Any]) -> dict[str, Any]:
    """Validate structure, citations, and numeric claims against supplied evidence."""
    items, limitations = payload.get("recommendations"), payload.get("limitations")
    if not isinstance(items, list) or not isinstance(limitations, list):
        raise ValueError("Invalid recommendation response structure")
    normalized = []
    numeric_values = set()
    for value in re.findall(NUMBER_PATTERN, json.dumps(source_catalog, ensure_ascii=False)):
        try: numeric_values.add(float(value.replace(",", "").rstrip("%")))
        except ValueError: pass
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ValueError("Recommendation item must contain text")
        refs = item.get("evidence_refs")
        if not isinstance(refs, list) or not refs or any(ref not in source_catalog for ref in refs):
            raise ValueError("Recommendation must cite supplied financial, forecast, root-cause, or policy evidence")
        for value in re.findall(NUMBER_PATTERN, item["text"]):
            number = float(value.replace(",", "").rstrip("%"))
            if not any(math.isclose(number, known, rel_tol=0.006, abs_tol=0.02) for known in numeric_values):
                raise ValueError(f"Unsupported numeric recommendation: {value}")
        normalized.append({"classification": "recommendation", "text": item["text"], "evidence_refs": refs})
    return {"recommendations": normalized, "limitations": [str(x) for x in limitations]}


def _profit_fallback(sections: dict[str, Any], source_catalog: dict[str, Any], reason: str = "") -> AgentResponse:
    """Offer cautious report-grounded next steps when Gemini is not configured."""
    financial = sections.get("financial_facts", {}).get("evidence", {})
    categories = financial.get("category_analysis", [])
    categories = [row for row in categories if isinstance(row, dict) and isinstance(row.get("profit_margin_pct"), (int, float))]
    categories.sort(key=lambda row: row["profit_margin_pct"], reverse=True)
    refs = [name for name in ("phase1", "phase2", "phase3") if name in source_catalog]
    if not refs:
        return AgentResponse("recommendation_agent", "insufficient_evidence", limitations=["No report evidence was available to ground next steps."])
    recommendations = []
    if categories:
        category = categories[0].get("Category", "the leading category")
        recommendations.append({
            "classification": "recommendation",
            "text": f"Review {category} as a candidate for a small, policy-reviewed sales test. It ranks highest on estimated margin in this sample; its COGS is synthetic, so replace it with verified supplier costs before changing pricing, inventory, or marketing spend.",
            "evidence_refs": ["phase1"],
        })
    recommendations.append({
        "classification": "recommendation",
        "text": "Capture actual product-level COGS and reconcile returned, cancelled, tax, and shipping amounts before judging whether a change improved real profit.",
        "evidence_refs": ["phase1"],
    })
    root = sections.get("root_cause_findings", {}).get("findings", [])
    if root and "phase2" in source_catalog:
        recommendations.append({
            "classification": "recommendation",
            "text": "Review the reported revenue changes and their associated contributors with the business owner before selecting a pricing or product action; the report shows association, not proven cause.",
            "evidence_refs": ["phase2"],
        })
    return AgentResponse(
        "recommendation_agent", "success", recommendations,
        {"evidence_refs": refs, "generator": "deterministic_fallback"},
        [{"source": "Phase 1–3 structured reports"}],
        [reason or "Gemini is not configured; these are cautious report-grounded suggestions, not AI-generated advice.",
         "Profit rankings use synthetic COGS estimates and are illustrative only."],
    )


class RecommendationAgent:
    name = "recommendation_agent"
    def run(self, request: AgentRequest) -> AgentResponse:
        dependencies = request.context.get("agent_responses", [])
        sections, source_catalog = build_recommendation_context(dependencies)
        if not source_catalog:
            return AgentResponse(self.name, "unavailable", limitations=["No successful agent evidence was available for recommendations."])
        load_local_env()
        asks_profit_action = bool(re.search(r"\bprofit(?:s|ability)?\b|\bmargin\b", request.question, re.IGNORECASE) and
                                  re.search(r"\b(improv\w*|increas\w*|gain\w*|grow\w*|boost\w*|maximi[sz]\w*|rais\w*|recommend\w*|should|how can|how to|what should)\b", request.question, re.IGNORECASE))
        if asks_profit_action and not (os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")):
            return _profit_fallback(sections, source_catalog)
        evidence_ids = list(source_catalog)
        prompt = (
            "You are a cautious financial recommendations analyst. Use only the supplied evidence. "
            "Do not invent facts, numbers, policies, causes, or future values. Root-cause contributions are "
            "associations, never proof of causation. Keep facts, findings, forecasts, and policy constraints "
            "in their own source sections; output recommendations only. Each recommendation must cite one "
            "or more exact evidence_refs from this list: " + json.dumps(evidence_ids) +
            ". Recommendations are suggestions for human review only; do not execute actions. Preserve policy "
            "limits and state when evidence is insufficient. If the question is how to improve profit but "
            "profit/cost data is unavailable, do not stop at that limitation: recommend practical, reversible "
            "actions grounded in observed revenue, order, quantity, product/category, forecast, and policy "
            "evidence. Frame them as sales or cost-to-serve levers, never claim a profit increase or quantify "
            "profit impact without actual product cost/COGS. If cost data is labeled synthetic or estimated, "
            "treat any profit values as illustrative only and say so. Mention the missing actual cost data as "
            "a measurement limitation.\n\n"
            "QUESTION:\n" + request.question + "\n\nEVIDENCE SECTIONS:\n" +
            json.dumps(sections, ensure_ascii=False, separators=(",", ":"))
        )
        try:
            output = validate_recommendations(_call_gemini(prompt, evidence_ids), source_catalog)
            status = "success" if output["recommendations"] else "insufficient_evidence"
            return AgentResponse(self.name, status, output["recommendations"],
                                 {"evidence_refs": evidence_ids}, [{"source": "Phase 1–5 structured reports and retrieved policy passages"}],
                                 output["limitations"])
        except Exception as exc:
            if asks_profit_action:
                return _profit_fallback(
                    sections, source_catalog,
                    f"Gemini recommendation synthesis was unavailable ({type(exc).__name__}); these are deterministic, report-grounded suggestions.",
                )
            return AgentResponse(self.name, "error", evidence={"evidence_refs": evidence_ids},
                                 limitations=[str(exc), "Recommendation generation failed; no actions were executed."])
