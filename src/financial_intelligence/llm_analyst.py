"""Evidence-grounded Phase 4 analyst using Gemini's generateContent REST API."""
from __future__ import annotations
import json
import math
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from .analyst_report import write_reports
from .prompt_builder import SECTIONS, CLASSIFICATIONS, build_analysis_context, build_prompt

NUMBER_PATTERN = r"(?<!\w)[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?!\w)"


def load_local_env(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE entries from .env without overriding process variables."""
    env_path = Path(path)
    if not env_path.is_file(): return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        key, value = line.split("=", 1); key = key.strip(); value = value.strip().strip("\"'")
        if key and key not in os.environ: os.environ[key] = value


def empty_analysis() -> dict[str, Any]:
    return {section: [] for section in SECTIONS}


def parse_response(response: str | dict[str, Any]) -> dict[str, Any]:
    """Parse either JSON text or a Gemini generateContent response envelope."""
    if isinstance(response, dict) and "candidates" in response:
        try:
            response = "".join(p.get("text", "") for p in response["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Gemini response did not contain generated text") from exc
    if isinstance(response, str):
        text = response.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
        try: response = json.loads(text)
        except json.JSONDecodeError as exc: raise ValueError("LLM response was not valid JSON") from exc
    if not isinstance(response, dict): raise ValueError("LLM response must be a JSON object")
    return response


def validate_analysis(payload: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Require all sections, valid item types, known evidence IDs, and evidence-backed numbers."""
    allowed_ids = set(context.get("evidence_ids", {}))
    result: dict[str, Any] = {}
    for section in SECTIONS:
        items = payload.get(section)
        if not isinstance(items, list): raise ValueError(f"Missing or invalid section: {section}")
        clean = []
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise ValueError(f"Invalid item in section: {section}")
            classification = item.get("classification")
            refs = item.get("evidence_ids")
            if classification not in CLASSIFICATIONS or not isinstance(refs, list) or not refs:
                raise ValueError(f"Classification and evidence citations are required in {section}")
            if any(ref not in allowed_ids for ref in refs): raise ValueError("Unknown evidence citation")
            # Numeric claims must match a number in the report context (within display rounding).
            # A statement may synthesize multiple report facts (for example, a
            # 60-month forecast history); keep its citations, but reject numbers
            # absent from the supplied Phase 1–3 evidence as a whole.
            available = _numbers_for_refs(context, list(allowed_ids))
            for token in re.findall(NUMBER_PATTERN, item["text"]):
                num = float(token.replace(",", "").rstrip("%"))
                if not any(math.isclose(num, n, rel_tol=0.006, abs_tol=0.02) for n in available):
                    raise ValueError(f"Unsupported numeric claim in {section}: {token}")
            clean.append({"text": item["text"], "classification": classification, "evidence_ids": refs})
        result[section] = clean
    return result


def _numbers_for_refs(context: dict[str, Any], refs: list[str]) -> list[float]:
    data = json.dumps(context, ensure_ascii=False)
    # Map each evidence source to the corresponding serialized report subtree.
    fragments = []
    p1, p2, p3 = context.get("phase1", {}), context.get("phase2", {}), context.get("phase3", {})
    for ref in refs:
        if ref.startswith("phase1_"):
            key = {"phase1_financial_summary": ("financial_summary", "kpis"), "phase1_data_quality": ("data_quality",),
                   "phase1_risk": ("risk_analysis",), "phase1_concentration": ("concentration_analysis",),
                   "phase1_trends": ("trend_analysis",)}.get(ref, ())
            fragments.extend(p1.get(k) for k in key)
        elif ref == "phase2_changes": fragments.append(p2.get("significant_changes"))
        elif ref == "phase3_forecasts": fragments.extend((p3.get("forecasts"), p3.get("assumptions")))
    text = json.dumps(fragments, ensure_ascii=False)
    number_tokens = re.findall(NUMBER_PATTERN, text)
    # Model identifiers such as arima_1_0_0_12 encode parameters as underscore-
    # separated digits; count these only as values present in the cited evidence.
    number_tokens.extend(re.findall(r"(?<=_)\d+(?=_|[^\w])", text))
    return [float(x.replace(",", "")) for x in number_tokens if _is_number(x)]


def _is_number(value: str) -> bool:
    try: return math.isfinite(float(value.replace(",", "")))
    except ValueError: return False


def call_gemini(prompt: str, api_key: str, model: str = "gemini-3.1-flash-lite", timeout: float = 45,
                base_url: str = "https://generativelanguage.googleapis.com/v1beta/models") -> dict[str, Any]:
    """Call Gemini JSON generation; key is sent only in a request header and never logged."""
    url = f"{base_url.rstrip('/')}/{model}:generateContent"
    item_schema = {"type": "OBJECT", "properties": {
        "text": {"type": "STRING"},
        "classification": {"type": "STRING", "enum": list(CLASSIFICATIONS)},
        "evidence_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
    }, "required": ["text", "classification", "evidence_ids"]}
    response_schema = {"type": "OBJECT", "properties": {
        section: {"type": "ARRAY", "items": item_schema} for section in SECTIONS
    }, "required": list(SECTIONS)}
    body = json.dumps({"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                       "generationConfig": {"responseMimeType": "application/json", "responseSchema": response_schema}}).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "x-goog-api-key": api_key}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            details = json.loads(exc.read().decode("utf-8", errors="replace"))
            message = details.get("error", {}).get("message", "")
        except (ValueError, AttributeError):
            message = ""
        # The API response body contains its diagnostic only; credentials are sent in
        # a request header and are never copied into this message.
        suffix = f": {message}" if message else ""
        raise RuntimeError(f"Gemini API returned HTTP {exc.code}{suffix}") from None
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError): raise TimeoutError("Gemini request timed out") from None
        raise RuntimeError(f"Gemini network error ({type(exc.reason).__name__}): {exc.reason}") from None


class LLMFinancialAnalyst:
    """Load Phase 1–3 outputs, request grounded analysis, and always write reports."""
    def __init__(self, output_dir: str | Path = "output", env_file: str | Path = ".env"):
        self.output_dir, self.env_file = Path(output_dir), Path(env_file)

    def run(self) -> dict[str, Any]:
        load_local_env(self.env_file)
        report = empty_analysis()
        report.update({"provider": os.getenv("LLM_PROVIDER", "gemini"), "model": os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL", "gemini-3.1-flash-lite"), "status": "unavailable"})
        try:
            paths = [self.output_dir / f for f in ("financial_report.json", "root_cause_report.json", "forecast_report.json")]
            if any(not p.is_file() for p in paths): raise FileNotFoundError("One or more Phase 1–3 JSON reports are missing")
            p1, p2, p3 = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
            context = build_analysis_context(p1, p2, p3)
            key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
            if not key:
                report["error"] = "GEMINI_API_KEY is not configured. Add it to .env to enable Gemini analysis."
                profit_assumption = p1.get("metadata", {}).get("feature_formulas", {}).get("profit", {}).get("assumption", "")
                profit_limitation = (
                    "Profit and margin are illustrative estimates from synthetic COGS assumptions; actual supplier/accounting costs are unavailable."
                    if "SYNTHETIC ESTIMATE" in str(profit_assumption)
                    else "Profit and cost are unavailable in the source dataset."
                )
                report["data_limitations"] = [
                    {"text": profit_limitation, "classification": "limitation", "evidence_ids": ["phase1_financial_summary"]},
                    {"text": report["error"], "classification": "limitation", "evidence_ids": ["phase1_data_quality"]},
                ]
            else:
                requested_model = report["model"]
                fallback_models = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", os.getenv("LLM_FALLBACK_MODELS", "gemini-3.5-flash-lite")).split(",") if m.strip()]
                models = list(dict.fromkeys([requested_model, *fallback_models]))
                last_error: Exception | None = None
                for model in models:
                    try:
                        raw = call_gemini(build_prompt(context), key, model,
                                          float(os.getenv("LLM_TIMEOUT_SECONDS", "45")),
                                          os.getenv("GEMINI_API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/models"))
                        validated = validate_analysis(parse_response(raw), context)
                        report.update(validated); report["model"] = model; report["status"] = "generated"
                        if model != requested_model:
                            report["fallback_from_model"] = requested_model
                        break
                    except (TimeoutError, RuntimeError, ValueError) as exc:
                        last_error = exc
                        # A quota/availability error can differ between models. Do not
                        # retry the same failed model, and stop on network/auth errors.
                        retryable = isinstance(exc, ValueError) or any(code in str(exc) for code in ("HTTP 429", "HTTP 404", "HTTP 500", "HTTP 503"))
                        if not retryable:
                            break
                if report["status"] != "generated" and last_error:
                    raise last_error
        except TimeoutError as exc:
            report["status"], report["error"] = "error", str(exc)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            report["status"], report["error"] = "error", str(exc)
        if 'p1' in locals():
            profit_assumption = p1.get("metadata", {}).get("feature_formulas", {}).get("profit", {}).get("assumption", "")
            if "SYNTHETIC ESTIMATE" in str(profit_assumption):
                note = "Profit and margin are illustrative estimates from synthetic COGS assumptions; actual supplier/accounting costs are unavailable."
                if not any(item.get("text") == note for item in report["data_limitations"] if isinstance(item, dict)):
                    report["data_limitations"].append({"text": note, "classification": "limitation", "evidence_ids": ["phase1_financial_summary"]})
        write_reports(report, self.output_dir)
        return report
