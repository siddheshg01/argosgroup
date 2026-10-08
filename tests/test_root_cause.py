import pandas as pd
from src.financial_intelligence.investigation import RootCauseInvestigator

def test_investigator_detects_change_and_writes_outputs(tmp_path):
    source=tmp_path/"sales.csv"
    pd.DataFrame({
        "OrderID":["1","2","3","4","5","6","7","8"],
        "OrderDate":["2024-01-01","2024-01-02","2024-01-03","2024-01-04","2024-01-05",
                      "2024-02-01","2024-02-02","2024-02-03"],
        "UnitPrice":[100]*8,"Quantity":[1]*8,"Discount":[0]*8,
        "ProductName":["A","A","A","B","B","A","C","C"],
        "Category":["X","X","X","Y","Y","X","Z","Z"],
        "City":["M","M","M","N","N","M","Q","Q"],
        "State":["S1","S1","S1","S2","S2","S1","S3","S3"],
        "CustomerID":["u1","u1","u1","u2","u2","u1","u3","u3"],
    }).to_csv(source,index=False)
    report=RootCauseInvestigator(source,tmp_path/"output",threshold_pct=20).run()
    assert report["metrics_investigated"] == ["revenue","quantity","orders"]
    assert report["significant_change_count"] == 3
    item=report["significant_changes"][0]
    assert item["percentage_change"] == -40
    assert item["contributors"]["product"]
    assert any("contributed to" in text for text in item["evidence"])
    assert (tmp_path/"output"/"root_cause_report.json").is_file()
    assert (tmp_path/"output"/"root_cause_summary.csv").is_file()
