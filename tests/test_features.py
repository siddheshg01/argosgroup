import pandas as pd
from src.financial_intelligence.schema import discover_schema
from src.financial_intelligence.features import engineer_features

def test_supported_revenue_and_margin_features():
    df=pd.DataFrame({"price":[10.0,20.0],"quantity":[2,1],"profit":[4.0,3.0],"date":["2024-01-02","2024-02-03"]})
    schema=discover_schema(df); out, info=engineer_features(df,schema)
    assert out["_fi_revenue"].tolist() == [20.0,20.0]
    assert round(out["_fi_profit_margin_pct"].iloc[0],1) == 20.0
    assert "month" in info["formulas"]

def test_amazon_camelcase_financial_fields():
    df=pd.DataFrame({"OrderID":["1"],"OrderDate":["2024-01-01"],"UnitPrice":[100.0],
                     "Quantity":[2],"Discount":[0.1],"Tax":[10.0],"ShippingCost":[3.0],"TotalAmount":[193.0]})
    schema=discover_schema(df)
    out, info=engineer_features(df,schema)
    assert schema["mapping"]["order_id"] == "OrderID"
    assert schema["mapping"]["order_amount"] == "TotalAmount"
    assert out["_fi_revenue"].iloc[0] == 180
    assert "discount rate" in info["formulas"]["revenue"]["formula"]
