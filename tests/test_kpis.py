import pandas as pd
from src.financial_intelligence.schema import discover_schema
from src.financial_intelligence.kpis import calculate_kpis

def test_supported_kpis_and_unavailable_refunds():
    df=pd.DataFrame({"order_id":["a","b"],"total_sales":[20,30],"profit":[5,10],"quantity":[2,3],"customer_id":["x","x"]})
    k=calculate_kpis(df,discover_schema(df))
    assert k["total_revenue"] == 50
    assert k["profit_margin_pct"] == 30
    assert k["number_of_customers"] == 1
    assert k["refund_status"] == "not_available_in_source_dataset"

