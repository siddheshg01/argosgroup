import pandas as pd
from src.financial_intelligence.schema import discover_schema
from src.financial_intelligence.validator import validate_dataset

def test_schema_and_quality_score():
    df=pd.DataFrame({"order_date":["2024-01-01","bad"],"total_sales":[2,-1],"quantity":[1,0]})
    schema=discover_schema(df); result=validate_dataset(df,schema)
    assert schema["mapping"]["revenue"] == "total_sales"
    assert result["invalid_dates"] == 1
    assert 0 <= result["data_quality_score"] <= 100

def test_duplicate_detection():
    df=pd.DataFrame({"total_sales":[5,5],"quantity":[1,1]})
    assert validate_dataset(df)["duplicate_records"] == 1

def test_validation_failures_reduce_quality_score():
    good=pd.DataFrame({"date":["2024-01-01"],"revenue":[10],"quantity":[1]})
    bad=pd.DataFrame({"date":["not-a-date"],"revenue":[-10],"quantity":[0]})
    assert validate_dataset(bad)["data_quality_score"] < validate_dataset(good)["data_quality_score"]
