"""Create financial features only from observed source measures."""
import pandas as pd
from .schema import discover_schema

def engineer_features(frame: pd.DataFrame, schema: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Add supported derived measures and return their formulas and sources."""
    schema = schema or discover_schema(frame)
    out = frame.copy()
    m = schema["mapping"]
    formulas = {}
    if "revenue" not in m and "price" in m and "quantity" in m:
        # UnitPrice is the unit list/selling price. Discount is a proportion when
        # all observed values are in [0, 1]. If its scale is ambiguous, do not
        # silently apply it.
        unit_sales = pd.to_numeric(out[m["price"]], errors="coerce") * pd.to_numeric(out[m["quantity"]], errors="coerce")
        formula = "price × quantity"
        sources = [m["price"], m["quantity"]]
        if "discount" in m:
            discount = pd.to_numeric(out[m["discount"]], errors="coerce")
            valid = discount.dropna()
            if valid.empty or ((valid >= 0) & (valid <= 1)).all():
                unit_sales = unit_sales * (1 - discount)
                formula = "price × quantity × (1 − discount rate)"
                sources.append(m["discount"])
        out["_fi_revenue"] = unit_sales
        m["revenue"] = "_fi_revenue"
        formulas["revenue"] = {"formula": formula, "sources": sources,
                                "assumption": "Derived merchandise sales; excludes tax and shipping. Discount is applied only when its observed scale is a fraction from 0 to 1; missing discounts remain missing. All order statuses are included because the dataset has no refund/reversal amount; consult order_status_counts for scope."}
    if "profit" not in m and "revenue" in m and "cost" in m:
        out["_fi_profit"] = pd.to_numeric(out[m["revenue"]], errors="coerce") - pd.to_numeric(out[m["cost"]], errors="coerce")
        m["profit"] = "_fi_profit"
        synthetic_cost = "CostDataType" in out and out["CostDataType"].astype(str).str.contains("synthetic", case=False).any()
        formulas["profit"] = {
            "formula": "revenue − estimated COGS" if synthetic_cost else "revenue − cost",
            "sources": [m["revenue"], m["cost"]],
            **({"assumption": "SYNTHETIC ESTIMATE: COGS is modeled from assumed product cost ratios, not actual accounting or supplier data. Profit and margin are illustrative only and must not be used as actual financial results."} if synthetic_cost else {}),
        }
    if "date" in m:
        dt = pd.to_datetime(out[m["date"]], errors="coerce")
        out["_fi_month"] = dt.dt.to_period("M").astype("string")
        out["_fi_quarter"] = dt.dt.to_period("Q").astype("string")
        out["_fi_year"] = dt.dt.year.astype("Int64")
        formulas.update({"month": {"formula": "month extracted from date", "sources": [m["date"]]},
                         "quarter": {"formula": "quarter extracted from date", "sources": [m["date"]]},
                         "year": {"formula": "year extracted from date", "sources": [m["date"]]}})
    if "revenue" in m and "profit" in m:
        rev = pd.to_numeric(out[m["revenue"]], errors="coerce").replace(0, pd.NA)
        out["_fi_profit_margin_pct"] = pd.to_numeric(out[m["profit"]], errors="coerce").div(rev).mul(100)
        formulas["profit_margin_pct"] = {"formula": "profit ÷ revenue × 100", "sources": [m["profit"], m["revenue"]]}
    if "revenue" in m and "quantity" in m:
        qty = pd.to_numeric(out[m["quantity"]], errors="coerce").replace(0, pd.NA)
        out["_fi_revenue_per_unit"] = pd.to_numeric(out[m["revenue"]], errors="coerce").div(qty)
        formulas["revenue_per_unit"] = {"formula": "revenue ÷ quantity", "sources": [m["revenue"], m["quantity"]]}
    if "order_amount" in m:
        formulas["order_amount"] = {"formula": "source-reported order total; no derivation", "sources": [m["order_amount"]],
                                     "note": "May include taxes, shipping, and non-delivered order statuses; not treated as net merchandise revenue."}
    schema["mapping"] = m
    schema["unavailable"] = [field for field in schema.get("unavailable", []) if field not in m]
    return out, {"formulas": formulas, "mapping": m}
