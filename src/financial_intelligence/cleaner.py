"""Conservative cleaning that records dropped rows instead of hiding them."""
import pandas as pd
from .schema import discover_schema

def clean_dataset(frame: pd.DataFrame, schema: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Normalize text/dates/numerics, remove exact duplicates, and quarantine invalid dates.

    Missing values are retained and reported; analytical code can skip unavailable
    measures without fabricating replacements.
    """
    schema = schema or discover_schema(frame)
    out = frame.copy()
    before = len(out)
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    for field in ("price", "quantity", "revenue", "order_amount", "profit", "cost", "discount", "tax", "shipping_cost"):
        col = schema["mapping"].get(field)
        if col:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    date_col = schema["mapping"].get("date")
    invalid_dates = 0
    if date_col:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    duplicated = int(out.duplicated().sum())
    out = out.drop_duplicates(keep="first")
    invalid_mask = out[date_col].isna() if date_col else pd.Series(False, index=out.index)
    invalid_dates = int(invalid_mask.sum())
    numeric_invalid = pd.Series(False, index=out.index)
    for field in ("price", "quantity", "revenue", "order_amount", "tax", "shipping_cost"):
        col = schema["mapping"].get(field)
        if not col:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        numeric_invalid |= (out[col].notna() & values.isna())
        if field in ("price", "quantity", "revenue", "order_amount"):
            numeric_invalid |= values.notna() & (values <= 0)
        elif field in ("tax", "shipping_cost"):
            numeric_invalid |= values.notna() & (values < 0)
    invalid_mask |= numeric_invalid
    invalid_records = int(invalid_mask.sum())
    quarantine = out.loc[invalid_mask].copy()
    out = out.loc[~invalid_mask].copy()
    return out, {"records_before": int(before), "records_after": int(len(out)),
                 "duplicates_removed": duplicated, "invalid_dates_quarantined": invalid_dates,
                 "invalid_numeric_records_quarantined": int(numeric_invalid.sum()),
                 "invalid_records_quarantined": invalid_records,
                 "missing_values_retained": {str(k): int(v) for k, v in out.isna().sum().items()},
                 "quarantined_records": quarantine}
