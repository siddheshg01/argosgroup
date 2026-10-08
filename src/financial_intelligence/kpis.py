"""Calculate dataset-supported summary metrics."""
import pandas as pd

def calculate_kpis(frame: pd.DataFrame, schema: dict) -> dict:
    """Return only KPIs supported by mapped source or engineered columns."""
    m = schema["mapping"]
    result = {}
    def numeric_total(field, key):
        col = m.get(field)
        if col and col in frame:
            result[key] = float(pd.to_numeric(frame[col], errors="coerce").sum())
    numeric_total("revenue", "total_revenue")
    numeric_total("order_amount", "total_order_amount")
    numeric_total("profit", "total_profit")
    numeric_total("quantity", "total_quantity")
    if "total_revenue" in result and "total_profit" in result:
        result["profit_margin_pct"] = result["total_profit"] / result["total_revenue"] * 100 if result["total_revenue"] else None
        synthetic_cost = "CostDataType" in frame and frame["CostDataType"].astype(str).str.contains("synthetic", case=False).any()
        if synthetic_cost:
            result["profit_status"] = "estimated_from_synthetic_cogs"
            result["profit_margin_status"] = "illustrative_estimate_only"
            result["cost_status"] = "synthetic_estimate_not_actual_cost"
    else:
        result["profit_status"] = "not_available_in_source_dataset"
        result["profit_margin_status"] = "not_available_without_profit_and_revenue"
    if "order_id" in m:
        result["total_orders"] = int(frame[m["order_id"]].nunique(dropna=True))
    if "customer_id" in m:
        result["number_of_customers"] = int(frame[m["customer_id"]].nunique(dropna=True))
    for field, key in (("product", "number_of_products"), ("category", "number_of_categories")):
        if field in m:
            result[key] = int(frame[m[field]].nunique(dropna=True))
    order_value_total = result.get("total_order_amount", result.get("total_revenue"))
    if order_value_total is not None and result.get("total_orders"):
        result["average_order_value"] = order_value_total / result["total_orders"]
    if "price" in m:
        result["average_selling_price"] = float(pd.to_numeric(frame[m["price"]], errors="coerce").mean())
    if "order_status" in m:
        result["order_status_counts"] = {str(k): int(v) for k, v in frame[m["order_status"]].value_counts(dropna=False).items()}
    for unsupported in ("cost", "refund", "discount"):
        if unsupported not in m:
            result[f"{unsupported}_status"] = "not_available_in_source_dataset"
    return result
