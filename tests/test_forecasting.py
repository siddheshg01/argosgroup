import numpy as np
import pandas as pd
import pytest

from src.financial_intelligence.forecasting import (
    _forecast_one_metric, chronological_split, prepare_monthly_series,
)
from src.financial_intelligence.forecast_models import (
    detect_annual_seasonality, evaluate_forecast, forecast_with_model, model_candidates,
)
from src.financial_intelligence.schema import discover_schema


def test_monthly_series_preparation_is_sorted_and_keeps_unobserved_gaps_missing():
    frame=pd.DataFrame({"OrderDate":["2024-03-01","2024-01-01","2024-01-02"],
                        "OrderID":["3","1","2"],"sales":[30,10,20],"qty":[1,2,3]})
    schema={"mapping":{"date":"OrderDate","order_id":"OrderID","revenue":"sales","quantity":"qty"}}
    monthly=prepare_monthly_series(frame,schema)
    assert list(monthly["revenue"].index.astype(str)) == ["2024-01","2024-02","2024-03"]
    assert monthly["revenue"].iloc[0] == 30
    assert pd.isna(monthly["revenue"].iloc[1])
    assert monthly["orders"].iloc[0] == 2


def test_chronological_split_has_no_shuffle_or_overlap():
    series=pd.Series(range(60),index=pd.period_range("2020-01",periods=60,freq="M"))
    parts=chronological_split(series)
    assert parts["train"].index.max() < parts["validation"].index.min()
    assert parts["validation"].index.max() < parts["test"].index.min()
    assert parts["train"].iloc[-1] < parts["validation"].iloc[0] < parts["test"].iloc[0]
    assert (len(parts["train"]),len(parts["validation"]),len(parts["test"])) == (36,12,12)


def test_split_rejects_missing_data_and_bad_ratios():
    with pytest.raises(ValueError): chronological_split([1,2,np.nan,4,5,6,7,8,9,10])
    with pytest.raises(ValueError): chronological_split(range(10),.8,.3)


def test_baseline_models_are_deterministic():
    values=[10,20,30,40]
    assert forecast_with_model(values,"naive",3)["values"].tolist() == [40,40,40]
    assert forecast_with_model(values,"moving_average_3",2)["values"].tolist() == [30,30]


def test_holt_and_arima_models_return_requested_horizon():
    values=np.linspace(10,30,36) + np.sin(np.arange(36))
    for name in ("holt_linear","arima_1_1_0"):
        forecast=forecast_with_model(values,name,4)
        assert len(forecast["values"]) == 4
        assert np.isfinite(forecast["values"]).all()
    seasonal=forecast_with_model(100+np.arange(48)*.2+10*np.sin(2*np.pi*np.arange(48)/12),
                                 "holt_winters_additive_12",3)
    assert len(seasonal["values"]) == 3


def test_seasonality_is_only_enabled_with_sufficient_training_evidence():
    seasonal=100+np.arange(60)*.2+15*np.sin(2*np.pi*np.arange(60)/12)
    assert detect_annual_seasonality(seasonal)["supported"]
    assert not detect_annual_seasonality(seasonal[:24])["supported"]
    assert any(model["name"] == "sarima_1_0_0_12" for model in model_candidates(True))
    assert all(not model["name"].startswith("sarima") for model in model_candidates(False))


def test_evaluation_metrics_and_mape_zero_handling():
    metrics=evaluate_forecast([0,100,200],[20,110,180])
    assert metrics["mae"] == pytest.approx(50/3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(300))
    assert metrics["mape_pct"] == pytest.approx(10)
    assert metrics["mape_observations"] == 2
    assert evaluate_forecast([0,0],[1,2])["mape_pct"] is None


def test_model_selection_uses_validation_and_produces_all_horizons():
    series=pd.Series(100+np.arange(24)*2+np.sin(np.arange(24)),
                     index=pd.period_range("2022-01",periods=24,freq="M"))
    result=_forecast_one_metric("revenue",series)
    assert result["status"] == "forecasted"
    assert result["selected_model"]["selection_basis"].startswith("lowest validation MAE")
    assert result["split"]["train"]["end"] < result["split"]["validation"]["start"]
    assert result["split"]["validation"]["end"] < result["split"]["test"]["start"]
    assert {"3_months","6_months","12_months"} == set(result["forecasts_by_horizon"])
    assert len(result["forecast_values"]) == 12
    assert len(result["prediction_intervals"]) == 12


def test_insufficient_series_is_reported_without_raising():
    series=pd.Series([1,2,3],index=pd.period_range("2024-01",periods=3,freq="M"))
    result=_forecast_one_metric("orders",series)
    assert result["status"] == "insufficient_data"
    assert result["forecast_values"] == []
