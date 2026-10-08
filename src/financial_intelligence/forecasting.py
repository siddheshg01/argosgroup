"""Phase 3 time-series preparation, temporal evaluation, and model selection."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .loader import load_dataset
from .schema import discover_schema
from .cleaner import clean_dataset
from .features import engineer_features
from .forecast_models import (detect_annual_seasonality, evaluate_forecast,
                              forecast_with_model, model_candidates)
from .forecast_report import write_forecast_outputs


def prepare_monthly_series(frame: pd.DataFrame, schema: dict) -> dict[str, pd.Series]:
    """Aggregate Phase 1 features chronologically and preserve uncovered months as NaN."""
    mapping = schema["mapping"]
    date_col = mapping.get("date")
    if not date_col:
        raise ValueError("Forecasting requires a recognized date field")
    work = frame.copy()
    work["_forecast_period"] = pd.to_datetime(work[date_col], errors="coerce").dt.to_period("M")
    work = work.dropna(subset=["_forecast_period"])
    if work.empty:
        return {}
    periods = pd.period_range(work["_forecast_period"].min(), work["_forecast_period"].max(), freq="M")
    result = {}
    for metric in ("revenue", "quantity"):
        column = mapping.get(metric)
        if column and column in work:
            values = pd.to_numeric(work[column], errors="coerce")
            result[metric] = values.groupby(work["_forecast_period"]).sum(min_count=1).reindex(periods).astype("float64")
    order_col = mapping.get("order_id")
    if order_col and order_col in work:
        result["orders"] = work.groupby("_forecast_period")[order_col].nunique(dropna=True).reindex(periods).astype("float64")
    return result


def chronological_split(series, train_ratio: float = 0.6, validation_ratio: float = 0.2) -> dict:
    """Split in time order into train, validation, and test without shuffling."""
    values = series if isinstance(series, pd.Series) else pd.Series(series)
    if not (0 < train_ratio < 1 and 0 < validation_ratio < 1 and train_ratio + validation_ratio < 1):
        raise ValueError("split ratios must be positive and leave a non-empty test set")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype="float64")).all():
        raise ValueError("time series has missing or non-finite values; do not impute silently")
    n = len(values)
    train_end = int(n * train_ratio)
    validation_end = int(n * (train_ratio + validation_ratio))
    if train_end < 3 or validation_end <= train_end or validation_end >= n:
        raise ValueError("insufficient observations for chronological train/validation/test splits")
    return {"train": values.iloc[:train_end].copy(),
            "validation": values.iloc[train_end:validation_end].copy(),
            "test": values.iloc[validation_end:].copy()}


def _forecast_values(values, non_negative=True):
    result = np.asarray(values, dtype="float64")
    return np.maximum(result, 0) if non_negative else result


def _forecast_one_metric(metric: str, series: pd.Series, threshold_pct: float = 5.0) -> dict:
    """Evaluate candidates on validation, reserve test for final evaluation, then refit."""
    base = {"metric": metric,
            "historical_period": {"start": str(series.index.min()) if len(series) else None,
                                  "end": str(series.index.max()) if len(series) else None,
                                  "observations": int(len(series))},
            "historical_values": [{"period": str(period), "value": float(value) if np.isfinite(value) else None}
                                  for period, value in series.items()]}
    if len(series) < 12:
        return {**base, "status": "insufficient_data", "reason": "at least 12 monthly observations are required",
                "models_tested": [], "selected_model": None, "forecast_values": [],
                "prediction_intervals": [], "significant_changes": [], "assumptions": []}
    if series.isna().any() or not np.isfinite(series.to_numpy(dtype="float64")).all():
        return {**base, "status": "invalid_history", "reason": "monthly series contains missing/non-finite periods; no imputation applied",
                "models_tested": [], "selected_model": None, "forecast_values": [],
                "prediction_intervals": [], "significant_changes": [], "assumptions": []}
    try:
        splits = chronological_split(series)
    except ValueError as exc:
        return {**base, "status": "insufficient_data", "reason": str(exc),
                "models_tested": [], "selected_model": None, "forecast_values": [],
                "prediction_intervals": [], "significant_changes": [], "assumptions": []}
    seasonality = detect_annual_seasonality(splits["train"].to_numpy())
    candidates = model_candidates(seasonality["supported"])
    comparisons = []
    validation_residuals = {}
    for candidate in candidates:
        name = candidate["name"]
        record = {"name": name, "parameters": candidate["parameters"], "status": "failed",
                  "validation_metrics": None, "test_metrics": None, "failure_reason": None}
        try:
            validation_prediction = _forecast_values(forecast_with_model(
                splits["train"], name, len(splits["validation"]), parameters=candidate["parameters"])["values"])
            record["validation_metrics"] = evaluate_forecast(splits["validation"].to_numpy(), validation_prediction)
            validation_residuals[name] = splits["validation"].to_numpy() - validation_prediction
            expanded_train = pd.concat([splits["train"], splits["validation"]])
            test_prediction = _forecast_values(forecast_with_model(
                expanded_train, name, len(splits["test"]), parameters=candidate["parameters"])["values"])
            record["test_metrics"] = evaluate_forecast(splits["test"].to_numpy(), test_prediction)
            record["status"] = "tested"
        except Exception as exc:  # one unstable candidate must not abort other models/metrics
            record["failure_reason"] = f"{type(exc).__name__}: {exc}"
        comparisons.append(record)
    eligible = [model for model in comparisons if model["status"] == "tested"
                and model["validation_metrics"] and model["validation_metrics"]["mae"] is not None]
    if not eligible:
        return {**base, "status": "model_failure", "reason": "all forecasting candidates failed",
                "seasonality_test": seasonality, "models_tested": comparisons, "selected_model": None,
                "forecast_values": [], "prediction_intervals": [], "significant_changes": [], "assumptions": []}
    selected = min(eligible, key=lambda item: (item["validation_metrics"]["mae"], item["name"]))
    name = selected["name"]
    candidate_params = next(model["parameters"] for model in candidates if model["name"] == name)
    final_fit = forecast_with_model(series, name, 12, parameters=candidate_params)
    point = _forecast_values(final_fit["values"])
    alpha = 0.05
    interval_method = "model-based ARIMA 95% prediction interval" if final_fit["lower"] is not None else "approximate 95% interval: selected-model validation RMSE × 1.96 × sqrt(horizon step)"
    if final_fit["lower"] is not None:
        lower = _forecast_values(final_fit["lower"])
        upper = _forecast_values(final_fit["upper"])
    else:
        residuals = validation_residuals[name]
        error_scale = float(np.sqrt(np.mean(residuals ** 2))) if len(residuals) else 0.0
        widening = 1.96 * error_scale * np.sqrt(np.arange(1, 13, dtype="float64"))
        lower, upper = np.maximum(0, point - widening), np.maximum(0, point + widening)
    future_periods = pd.period_range(series.index[-1] + 1, periods=12, freq="M")
    forecast_values = [{"period": str(period), "value": float(value)} for period, value in zip(future_periods, point)]
    intervals = [{"period": str(period), "lower": float(lo), "upper": float(hi), "level": 0.95,
                  "method": interval_method} for period, lo, hi in zip(future_periods, lower, upper)]
    horizons = {}
    significant = []
    latest = float(series.iloc[-1])
    for horizon in (3, 6, 12):
        endpoint = float(point[horizon - 1])
        change = ((endpoint - latest) / latest * 100) if latest != 0 else None
        horizons[f"{horizon}_months"] = {"period": str(future_periods[horizon - 1]),
                                          "forecast_endpoint": endpoint,
                                          "forecast_total": float(np.sum(point[:horizon]))}
        if change is not None and abs(change) >= threshold_pct:
            significant.append({"horizon_months": horizon, "period": str(future_periods[horizon - 1]),
                                "expected_change_pct": float(change),
                                "direction": "increase" if change > 0 else "decrease",
                                "threshold_pct": threshold_pct})
    first_quarter, last_quarter = float(np.mean(point[:3])), float(np.mean(point[-3:]))
    trend_pct = ((last_quarter - first_quarter) / abs(first_quarter) * 100) if first_quarter else 0.0
    trend = "Increasing" if trend_pct > 2 else "Decreasing" if trend_pct < -2 else "Stable"
    historical_mean = float(series.iloc[-12:].mean())
    forecast_mean = float(np.mean(point))
    comparison_pct = ((forecast_mean - historical_mean) / historical_mean * 100) if historical_mean else None
    return {
        **base, "status": "forecasted", "seasonality_test": seasonality,
        "split": {key: {"start": str(value.index[0]), "end": str(value.index[-1]), "observations": int(len(value))}
                  for key, value in splits.items()},
        "models_tested": comparisons,
        "selected_model": {"name": name, "parameters": final_fit["parameters"],
                           "selection_basis": "lowest validation MAE; test segment was not used for selection"},
        "validation_metrics": selected["validation_metrics"], "test_metrics": selected["test_metrics"],
        "forecast_period": {"start": str(future_periods[0]), "end": str(future_periods[-1]), "months": 12},
        "forecast_values": forecast_values, "forecasts_by_horizon": horizons,
        "prediction_intervals": intervals,
        "trend": {"direction": trend, "change_between_first_and_last_forecast_quarter_pct": float(trend_pct),
                  "stable_band_pct": 2.0},
        "historical_vs_forecast": {"historical_last_12_month_mean": historical_mean,
                                   "forecast_next_12_month_mean": forecast_mean,
                                   "mean_change_pct": comparison_pct},
        "significant_changes": significant,
        "assumptions": ["Chronological 60/20/20 split; no shuffling.",
                        "Model selected using validation MAE only; test data is held out from selection.",
                        "Annual seasonality is enabled only when the training-only detrended lag-12 test clears its documented threshold.",
                        "Forecasts are non-negative because the metrics are sales/revenue/count aggregates.",
                        "Orders count unique source order IDs; revenue uses the Phase 1 merchandise-revenue feature.",
                        "Approximate intervals use validation errors for models without native prediction intervals; they are not guarantees.",
                        "Forecasts describe the supplied history and do not adjust for external business events."],
        "interval_method": interval_method,
    }


class FinancialForecaster:
    """Prepare Phase 1 monthly data, compare models, and write Phase 3 outputs."""
    def __init__(self, dataset_path="data/raw/amazon_sales_2024.csv", output_dir="output",
                 significant_change_pct: float = 5.0):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.significant_change_pct = float(significant_change_pct)
        if self.significant_change_pct < 0:
            raise ValueError("significant_change_pct must be non-negative")

    def run(self) -> dict:
        """Reuse Phase 1 ingestion/features and generate forecasts for available metrics."""
        raw = load_dataset(self.dataset_path)
        schema = discover_schema(raw)
        cleaned, cleaning = clean_dataset(raw, schema)
        featured, feature_info = engineer_features(cleaned, schema)
        series = prepare_monthly_series(featured, schema)
        results = [_forecast_one_metric(metric, values, self.significant_change_pct)
                   for metric, values in series.items() if metric in ("revenue", "quantity", "orders")]
        successful = [item for item in results if item.get("status") == "forecasted"]
        model_names = sorted({model["name"] for item in successful for model in item["models_tested"]
                              if model["status"] == "tested"})
        report = {
            "metadata": {"phase": "Phase 3 - Financial Forecasting Engine",
                         "dataset": self.dataset_path.name, "records_analyzed": int(len(featured)),
                         "monthly_periods_available": int(max((len(item) for item in series.values()), default=0)),
                         "cleaning_summary": {key: value for key, value in cleaning.items() if key != "quarantined_records"},
                         "revenue_formula": feature_info.get("formulas", {}).get("revenue")},
            "metrics_forecasted": [item["metric"] for item in successful],
            "models_tested": model_names,
            "forecasts": results,
            "assumptions": ["Profit is excluded because Phase 1 found no supported profit/cost fields.",
                            "Forecast results are conditional on historical patterns and are not guarantees."],
        }
        write_forecast_outputs(report, self.output_dir)
        return report

