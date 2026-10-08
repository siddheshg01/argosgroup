from src.financial_intelligence.risk import calculate_risk

def test_risk_score_and_level():
    trends={"direction":{"revenue":"decreased","profit":"decreased"}}
    result=calculate_risk(trends,{"top_product_revenue_share_pct":65},{"data_quality_score":55})
    assert result["risk_score"] == 75
    assert result["risk_level"] == "HIGH"
    assert 0 <= result["risk_score"] <= 100

