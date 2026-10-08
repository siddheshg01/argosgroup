"""Describe historical monthly changes without forecasting."""
import pandas as pd

def analyze_trends(frame: pd.DataFrame, schema: dict) -> dict | None:
    """Aggregate supported measures by calendar month and calculate MoM change."""
    m = schema["mapping"]
    if "date" not in m: return None
    work = frame.copy()
    work["_month"] = pd.to_datetime(work[m["date"]], errors="coerce").dt.to_period("M").astype("string")
    specs = {out: (m[field], "sum") for field,out in (("revenue","revenue"),("profit","profit"),("quantity","quantity")) if field in m}
    if "order_id" in m: specs["orders"] = (m["order_id"], "nunique")
    if not specs: return None
    monthly = work.dropna(subset=["_month"]).groupby("_month").agg(**{k: pd.NamedAgg(column=c, aggfunc=a) for k,(c,a) in specs.items()}).sort_index()
    if "profit" in monthly and "revenue" in monthly:
        monthly["profit_margin_pct"] = monthly["profit"].div(monthly["revenue"].replace(0, pd.NA)).mul(100)
    trends = monthly.copy()
    for metric in ("revenue", "profit", "quantity", "profit_margin_pct"):
        if metric in trends:
            trends[f"{metric}_growth_pct"] = trends[metric].pct_change().replace([float("inf"), -float("inf")], pd.NA) * 100
    records = trends.reset_index().where(pd.notna(trends.reset_index()), None).to_dict(orient="records")
    direction = {}
    for metric in ("revenue", "profit", "quantity", "profit_margin_pct"):
        if metric in monthly and len(monthly) >= 2:
            a,b=monthly[metric].iloc[0],monthly[metric].iloc[-1]
            direction[metric] = "increased" if b>a else "decreased" if b<a else "stable"
    return {"monthly": records, "direction": direction,
            "direction_comparison": "first observed month compared with last observed month; descriptive only"}
