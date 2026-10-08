"""Group supported measures by geography, products, categories, and customers."""
import pandas as pd

def _group(frame: pd.DataFrame, key: str, schema: dict) -> list[dict] | None:
    m = schema["mapping"]
    col = m.get(key)
    if not col or col not in frame: return None
    metrics = {}
    for field, output in (("revenue", "revenue"), ("profit", "profit"), ("quantity", "quantity")):
        source = m.get(field)
        if source and source in frame: metrics[output] = (source, "sum")
    if "order_id" in m: metrics["orders"] = (m["order_id"], "nunique")
    if not metrics: return None
    result = frame.groupby(col, dropna=False).agg(**{name: pd.NamedAgg(column=source, aggfunc=func) for name,(source,func) in metrics.items()}).reset_index()
    if "revenue" in result and "profit" in result:
        result["profit_margin_pct"] = result["profit"].div(result["revenue"].replace(0, pd.NA)).mul(100)
    if "revenue" in result and "orders" in result:
        result["average_order_value"] = result["revenue"].div(result["orders"].replace(0, pd.NA))
    return result.sort_values("revenue", ascending=False, na_position="last").head(100).to_dict(orient="records") if "revenue" in result else result.to_dict(orient="records")

def analyze_segments(frame: pd.DataFrame, schema: dict) -> dict:
    """Return supported segments and pseudonymized customer leaders."""
    m = schema["mapping"]
    result = {"location_analysis": _group(frame, "location", schema),
              "regional_analysis": _group(frame, "region", schema),
              "product_analysis": _group(frame, "product", schema),
              "category_analysis": _group(frame, "category", schema)}
    customer_col = m.get("customer_id")
    customers = None
    if customer_col:
        agg = {"orders": pd.NamedAgg(column=m["order_id"], aggfunc="nunique")} if "order_id" in m else {}
        for field in ("revenue", "profit", "quantity"):
            if field in m: agg[field] = pd.NamedAgg(column=m[field], aggfunc="sum")
        if agg:
            table = frame.groupby(customer_col, dropna=False).agg(**agg).reset_index()
            # Stable pseudonyms avoid exposing customer IDs or names.
            ids = sorted(table[customer_col].dropna().astype(str).unique())
            names = {value: f"Customer {i:05d}" for i, value in enumerate(ids, 1)}
            table[customer_col] = table[customer_col].astype(str).map(names).fillna("Customer unknown")
            customer_records = table.to_dict(orient="records")
            by_revenue = sorted(customer_records, key=lambda row: row.get("revenue", 0), reverse=True)[:100]
            by_orders = sorted(customer_records, key=lambda row: row.get("orders", 0), reverse=True)[:100]
            customers = {"total_customers": int(frame[customer_col].nunique(dropna=True)),
                         "top_revenue_customers": by_revenue,
                         "top_order_frequency_customers": by_orders}
    result["customer_analysis"] = customers
    date_col, revenue_col, product_col = m.get("date"), m.get("revenue"), m.get("product")
    if date_col and revenue_col and product_col:
        work = frame[[date_col, revenue_col, product_col]].copy()
        work["_month"] = pd.to_datetime(work[date_col], errors="coerce").dt.to_period("M").astype("string")
        monthly = work.dropna(subset=["_month"]).groupby([product_col, "_month"])[revenue_col].sum().unstack()
        months = sorted(monthly.columns.dropna())
        if len(months) >= 2:
            prev, latest = months[-2], months[-1]
            comparable = monthly[[prev, latest]].dropna()
            comparable = comparable[comparable[prev] != 0].copy()
            comparable["change_pct"] = (comparable[latest] - comparable[prev]) / comparable[prev] * 100
            declining = comparable[comparable["change_pct"] < 0].sort_values("change_pct").head(100)
            result["product_trend_analysis"] = {
                "comparison": f"{prev} vs {latest}; products must have sales in both months",
                "declining_products": [{"product": str(index), "previous_month_revenue": float(row[prev]),
                                        "latest_month_revenue": float(row[latest]), "change_pct": float(row["change_pct"])}
                                       for index, row in declining.iterrows()],
            }
        else:
            result["product_trend_analysis"] = None
    status_col = m.get("order_status")
    if status_col:
        grouped = frame.groupby(status_col, dropna=False).size().rename("records").reset_index()
        result["order_status_analysis"] = grouped.to_dict(orient="records")
    return result

def concentration_analysis(frame: pd.DataFrame, schema: dict) -> dict:
    """Report revenue share for largest product/category/location when available."""
    m = schema["mapping"]
    rev = m.get("revenue")
    if not rev or rev not in frame: return {}
    total = pd.to_numeric(frame[rev], errors="coerce").sum()
    result = {}
    if not total: return result
    for field, label in (("product", "product"), ("category", "category"), ("location", "location")):
        col = m.get(field)
        if col:
            shares = pd.to_numeric(frame[rev], errors="coerce").groupby(frame[col]).sum().sort_values(ascending=False) / total * 100
            result[f"top_{label}_revenue_share_pct"] = float(shares.iloc[0]) if len(shares) else None
            if field == "product": result["top_5_products_revenue_share_pct"] = float(shares.head(5).sum())
    return result
