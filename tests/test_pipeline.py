import json
import pandas as pd
from src.financial_intelligence.pipeline import FinancialIntelligencePipeline

def test_pipeline_writes_report_summary_and_supported_charts(tmp_path):
    raw=tmp_path/"amazon.csv"
    pd.DataFrame({
        "OrderID":["a","b","c","d"],
        "OrderDate":["2024-01-02","2024-02-02","2024-02-20","2024-03-02"],
        "CustomerID":["x","y","x","z"],"ProductName":["A","B","A","B"],
        "Category":["cat1","cat2","cat1","cat2"],"City":["X","Y","X","Z"],
        "State":["S1","S2","S1","S3"],"UnitPrice":[100,200,100,200],
        "Quantity":[1,1,2,1],"Discount":[0,0.1,0,0],"Tax":[0,0,0,0],
        "ShippingCost":[0,0,0,0],"TotalAmount":[100,180,200,200],"OrderStatus":["Delivered"]*4
    }).to_csv(raw,index=False)
    output=tmp_path/"output"
    report=FinancialIntelligencePipeline(raw,output).run()
    assert report["metadata"]["records_analyzed"] == 4
    assert report["kpis"]["total_orders"] == 4
    assert report["kpis"]["total_revenue"] == 680
    assert report["kpis"]["total_order_amount"] == 680
    assert "top_category_revenue_share_pct" in report["concentration_analysis"]
    assert len(report["trend_analysis"]["monthly"]) == 3
    assert json.loads((output/"financial_report.json").read_text(encoding="utf-8"))["metadata"]["records_analyzed"] == 4
    assert (output/"financial_summary.csv").is_file()
    assert (output/"charts"/"monthly_revenue.png").is_file()
    assert (output/"charts"/"revenue_by_product.png").is_file()
