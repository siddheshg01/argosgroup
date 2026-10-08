"""Forecasting model definitions and evaluation helpers for Phase 3."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def model_candidates(seasonal_supported: bool) -> list[dict]:
    """List deterministic baseline and statistical model specifications."""
    candidates = [
        {"name": "naive", "parameters": {"method": "last_observation"}},
        {"name": "moving_average_3", "parameters": {"window": 3}},
        {"name": "holt_linear", "parameters": {"trend": "additive", "seasonal": None}},
        {"name": "arima_1_1_0", "parameters": {"order": [1, 1, 0]}},
        {"name": "arima_0_1_1", "parameters": {"order": [0, 1, 1]}},
    ]
    if seasonal_supported:
        candidates.append({"name": "holt_winters_additive_12",
                           "parameters": {"trend": "additive", "seasonal": "additive", "seasonal_periods": 12}})
        candidates.append({"name": "sarima_1_0_0_12",
                           "parameters": {"order": [1, 0, 0], "seasonal_order": [1, 0, 0, 12]}})
    return candidates


def forecast_with_model(values, model_name: str, horizon: int, *, parameters: dict | None = None,
                        alpha: float = 0.05) -> dict:
    """Fit one candidate and return point forecasts plus optional model intervals."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    series = pd.Series(values, dtype="float64")
    if len(series) == 0 or not np.isfinite(series.to_numpy()).all():
        raise ValueError("model input must be non-empty and finite")
    params = parameters or {}
    if model_name == "naive":
        return {"values": np.repeat(series.iloc[-1], horizon), "lower": None, "upper": None,
                "parameters": {"method": "last_observation"}}
    if model_name == "moving_average_3":
        window = min(3, len(series))
        return {"values": np.repeat(series.iloc[-window:].mean(), horizon), "lower": None, "upper": None,
                "parameters": {"window": window}}
    if model_name in ("holt_linear", "holt_winters_additive_12"):
        seasonal = "add" if model_name == "holt_winters_additive_12" else None
        kwargs = {"trend": "add", "seasonal": seasonal, "initialization_method": "estimated"}
        if seasonal:
            kwargs["seasonal_periods"] = 12
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ExponentialSmoothing(series, **kwargs).fit(optimized=True)
        forecasts = np.asarray(fitted.forecast(horizon), dtype="float64")
        return {"values": forecasts, "lower": None, "upper": None,
                "parameters": params or {"trend": "add", "seasonal": seasonal,
                                          "seasonal_periods": 12 if seasonal else None}}
    if model_name.startswith("arima_") or model_name.startswith("sarima_"):
        order = tuple(params.get("order", [1, 1, 0]))
        seasonal_order = tuple(params.get("seasonal_order", [0, 0, 0, 0]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ARIMA(series, order=order, seasonal_order=seasonal_order,
                           enforce_stationarity=False, enforce_invertibility=False).fit()
        result = fitted.get_forecast(steps=horizon)
        confidence = np.asarray(result.conf_int(alpha=alpha), dtype="float64")
        return {"values": np.asarray(result.predicted_mean, dtype="float64"),
                "lower": confidence[:, 0], "upper": confidence[:, 1],
                "parameters": {"order": list(order), "seasonal_order": list(seasonal_order)}}
    raise ValueError(f"Unknown forecasting model: {model_name}")


def evaluate_forecast(actual, predicted) -> dict:
    """Calculate MAE/RMSE and MAPE over non-zero actuals only."""
    actual_values = np.asarray(actual, dtype="float64").reshape(-1)
    predicted_values = np.asarray(predicted, dtype="float64").reshape(-1)
    if len(actual_values) != len(predicted_values):
        raise ValueError("actual and predicted must have the same length")
    valid = np.isfinite(actual_values) & np.isfinite(predicted_values)
    actual_values, predicted_values = actual_values[valid], predicted_values[valid]
    if len(actual_values) == 0:
        return {"mae": None, "rmse": None, "mape_pct": None, "observations": 0, "mape_observations": 0}
    errors = predicted_values - actual_values
    nonzero = actual_values != 0
    mape = (float(np.mean(np.abs(errors[nonzero] / actual_values[nonzero])) * 100)
            if nonzero.any() else None)
    return {"mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mape_pct": mape, "observations": int(len(errors)),
            "mape_observations": int(nonzero.sum())}


def detect_annual_seasonality(values, minimum_strength: float | None = None) -> dict:
    """Test lag-12 residual correlation after removing a linear trend.

    Annual seasonality is used only with at least 36 finite monthly observations
    and a lag-12 residual correlation above both 0.30 and an approximate
    95%-noise threshold (2/sqrt(number of lagged pairs)).
    """
    series = np.asarray(values, dtype="float64")
    series = series[np.isfinite(series)]
    if len(series) < 36:
        return {"supported": False, "lag": 12, "strength": None,
                "reason": "fewer than 36 monthly observations"}
    x = np.arange(len(series), dtype="float64")
    detrended = series - np.polyval(np.polyfit(x, series, 1), x)
    left, right = detrended[:-12], detrended[12:]
    if np.std(left) == 0 or np.std(right) == 0:
        strength = 0.0
    else:
        strength = float(np.corrcoef(left, right)[0, 1])
    threshold = max(0.30, minimum_strength or 0.0, 2 / np.sqrt(len(left)))
    supported = bool(np.isfinite(strength) and strength >= threshold)
    return {"supported": supported, "lag": 12, "strength": strength,
            "threshold": float(threshold),
            "reason": "lag-12 detrended correlation clears threshold" if supported else "lag-12 evidence did not clear threshold"}
