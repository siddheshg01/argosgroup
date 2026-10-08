import pandas as pd
import pytest
from src.financial_intelligence.loader import load_dataset

def test_load_dataset_and_inspection(tmp_path):
    path=tmp_path/"data.csv"; pd.DataFrame({"amount":[2,3]}).to_csv(path,index=False)
    df=load_dataset(path)
    assert df.shape == (2,1)
    assert df.attrs["inspection"]["columns"] == ["amount"]

def test_missing_file_raises():
    with pytest.raises(FileNotFoundError): load_dataset("missing.csv")

