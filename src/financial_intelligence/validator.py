"""Reusable, explainable checks for the fields present in a dataset."""
import pandas as pd
from .schema import discover_schema

def validate_dataset(frame: pd.DataFrame, schema: dict | None = None) -> dict:
    """Report completeness, duplicates, invalid values, and a bounded quality score.

    Score starts at 100; subtracts up to 40 for missing cells by percentage,
    20 for duplicate rows by percentage, up to 5 proportional points for
    invalid dates and for each invalid numeric field. Missing date/revenue/quantity
    concepts each cost 5 points. The score is an internal indicator, not a
    financial or industry-standard score.
    """
    schema = schema or discover_schema(frame)
    mapping = schema["mapping"]
    n = max(len(frame), 1)
    missing = {str(k): int(v) for k, v in frame.isna().sum().items()}
    duplicate_rows = int(frame.duplicated().sum())
    invalid_dates = 0
    if "date" in mapping:
        parsed_dates = pd.to_datetime(frame[mapping["date"]], errors="coerce")
        invalid_dates = int((parsed_dates.isna() & frame[mapping["date"]].notna()).sum())
    invalid_numeric = {}
    for field in ("price", "quantity", "revenue", "order_amount", "tax", "shipping_cost"):
        col = mapping.get(field)
        if col:
            values = pd.to_numeric(frame[col], errors="coerce")
            bad = values.isna() & frame[col].notna()
            if field in ("price", "quantity", "revenue", "order_amount"):
                bad |= values <= 0
            invalid_numeric[field] = int(bad.sum())
    revenue_derivable = (("price" in mapping and "quantity" in mapping)
                         or ("order_amount" in mapping and "tax" in mapping and "shipping_cost" in mapping))
    schema_issues = [f"Missing expected concept: {f}" for f in ("date", "quantity") if f not in mapping]
    if "revenue" not in mapping and not revenue_derivable:
        schema_issues.append("Revenue is neither present nor derivable from supported source fields")
    missing_pct = sum(missing.values()) / (len(frame.columns) * n) * 100 if len(frame.columns) else 100
    score = 100 - min(40, missing_pct * 0.4) - min(20, duplicate_rows / n * 100 * 0.2)
    score -= min(5, invalid_dates / n * 5)
    score -= sum(min(5, count / n * 5) for count in invalid_numeric.values())
    categorical_summary = {
        str(col): {"unique_values": int(frame[col].nunique(dropna=True)),
                   "top_values": {str(k): int(v) for k, v in frame[col].value_counts(dropna=True).head(10).items()}}
        for col in frame.select_dtypes(include=["object", "string", "category"]).columns
    }
    score -= 5 * len(schema_issues)
    return {
        "total_records": int(len(frame)), "total_columns": int(len(frame.columns)),
        "missing_values": missing, "duplicate_records": duplicate_rows,
        "duplicate_ids": int(frame[mapping["order_id"]].duplicated().sum()) if "order_id" in mapping else None,
        "invalid_dates": invalid_dates, "invalid_numeric_values": invalid_numeric,
        "categorical_summary": categorical_summary,
        "schema_issues": schema_issues, "data_quality_score": round(max(0.0, min(100.0, score)), 2),
        "score_method": "100 minus min(40, missing-cell percentage × 0.4), min(20, exact-duplicate-row percentage × 0.2), min(5, invalid-date fraction × 5), the same capped penalty per invalid numeric field, and 5 per missing/unavailable core concept; clamped to 0–100.",
    }
