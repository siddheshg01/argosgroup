"""Build and serialize a JSON-compatible Phase 1 report."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

def _safe(value):
    if isinstance(value, dict): return {str(k): _safe(v) for k,v in value.items() if k != "quarantined_records"}
    if isinstance(value, (list, tuple)): return [_safe(v) for v in value]
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)): return None
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (pd.Timestamp,)): return value.isoformat()
    return value

def build_report(schema: dict, quality: dict, cleaning: dict, kpis: dict, segments: dict,
                 trends: dict | None, concentration: dict, risk: dict, records: int,
                 formulas: dict | None = None, inspection: dict | None = None) -> dict:
    """Combine pipeline outputs under stable keys for downstream consumers."""
    report = {"metadata": {"dataset": "Amazon Sales & Trading Insights Dataset 2024",
                           "phase": "Phase 1 - Financial Intelligence Engine", "records_analyzed": records,
                           "schema_mapping": schema["mapping"], "unavailable_fields": schema["unavailable"],
                           "feature_formulas": formulas or {}, "dataset_inspection": inspection or {}},
              "data_quality": quality, "cleaning_summary": {k:v for k,v in cleaning.items() if k != "quarantined_records"},
              "financial_summary": {k:v for k,v in kpis.items() if k.startswith("total_") or k == "profit_margin_pct"},
              "kpis": kpis, **segments, "trend_analysis": trends,
              "concentration_analysis": concentration, "risk_analysis": risk}
    return _safe(report)

def save_report(report: dict, path: str | Path) -> None:
    """Write indented UTF-8 JSON to a destination, creating parent folders."""
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_safe(report), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
