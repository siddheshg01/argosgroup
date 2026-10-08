"""A transparent descriptive business-health indicator, not a risk model."""
def calculate_risk(trends: dict | None, concentration: dict, quality: dict) -> dict:
    """Score declines, concentration, and data quality using fixed additive rules.

    Each supported decline (revenue/profit/quantity/profit margin) adds 15;
    top product/category/location share above 50% adds 10; quality below 80 adds 20, below 60
    adds 35. Total is capped at 100. This is an internal heuristic only.
    """
    score, factors = 0, []
    if trends:
        for metric, label in (("revenue", "Revenue"), ("profit", "Profit"),
                              ("profit_margin_pct", "Profit margin"), ("quantity", "Sales volume")):
            if trends.get("direction", {}).get(metric) == "decreased":
                score += 15; factors.append(f"{label} was lower in the last observed month than in the first")
    for key,label in (("top_product_revenue_share_pct","Product"),
                      ("top_category_revenue_share_pct","Category"),
                      ("top_location_revenue_share_pct","Location")):
        if concentration.get(key, 0) > 50:
            score += 10; factors.append(f"Revenue is concentrated in one {label.lower()} ({concentration[key]:.1f}%)")
    q = quality.get("data_quality_score", 100)
    if q < 60: score += 35; factors.append("Data quality score is below 60")
    elif q < 80: score += 20; factors.append("Data quality score is below 80")
    score = min(100, score)
    level = "LOW" if score <= 25 else "MEDIUM" if score <= 50 else "HIGH" if score <= 75 else "CRITICAL"
    return {"risk_score": score, "risk_level": level, "risk_factors": factors,
            "scoring_method": "15 per first-to-last observed decline in revenue/profit/profit margin/quantity; 10 per top product/category/location share above 50%; 20 for quality below 80 or 35 below 60; capped at 100."}
