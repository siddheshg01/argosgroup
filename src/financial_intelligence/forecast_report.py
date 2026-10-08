"""JSON, CSV, and chart outputs for Phase 3 forecasting."""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe(value):
    if isinstance(value, dict): return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_safe(v) for v in value]
    if value is None: return None
    if isinstance(value, (float, int)): return value if math.isfinite(value) else None
    if hasattr(value, "item"): return _safe(value.item())
    return value


def _make_charts(report: dict, chart_dir: Path) -> int:
    chart_dir.mkdir(parents=True, exist_ok=True)
    forecasted = [item for item in report["forecasts"] if item.get("status") == "forecasted"]
    made = 0
    for item in forecasted:
        historical = item["historical_values"][-36:]
        future = item["forecast_values"]
        intervals = item["prediction_intervals"]
        hist_dates = [pd.Period(row["period"]).to_timestamp() for row in historical]
        future_dates = [pd.Period(row["period"]).to_timestamp() for row in future]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hist_dates, [row["value"] for row in historical], label="Historical", color="#555b66")
        ax.plot(future_dates, [row["value"] for row in future], label="Forecast", color="#872aab", marker="o", markersize=3)
        ax.fill_between(future_dates, [row["lower"] for row in intervals],
                        [row["upper"] for row in intervals], color="#872aab", alpha=.16, label="95% interval")
        ax.axvline(hist_dates[-1], color="#999", linestyle="--", linewidth=1)
        ax.set(title=f"{item['metric'].title()}: history and 12-month forecast",
               xlabel="Month", ylabel=item["metric"].title())
        ax.legend(); fig.tight_layout()
        fig.savefig(chart_dir / f"{item['metric']}_historical_vs_forecast.png"); plt.close(fig)
        made += 1

    # A shared figure compares validation MAE in separate metric panels, so unlike
    # raw-scale combined bars it remains readable across unlike units.
    comparable = [item for item in forecasted if item.get("models_tested")]
    if comparable:
        fig, axes = plt.subplots(1, len(comparable), figsize=(max(7, 5 * len(comparable)), 5), squeeze=False)
        for ax, item in zip(axes[0], comparable):
            models = [model for model in item["models_tested"] if model["validation_metrics"] and model["validation_metrics"]["mae"] is not None]
            ax.bar([model["name"] for model in models], [model["validation_metrics"]["mae"] for model in models], color="#872aab")
            ax.set_title(item["metric"].title()); ax.set_ylabel("Validation MAE"); ax.tick_params(axis="x", rotation=65)
        fig.suptitle("Forecast model comparison (validation MAE)"); fig.tight_layout()
        fig.savefig(chart_dir / "model_comparison.png"); plt.close(fig)
        made += 1

        fig, axes = plt.subplots(len(comparable), 1, figsize=(10, 3.5 * len(comparable)), squeeze=False)
        for ax, item in zip(axes[:, 0], comparable):
            dates = [pd.Period(row["period"]).to_timestamp() for row in item["forecast_values"]]
            ax.plot(dates, [row["value"] for row in item["forecast_values"]], color="#872aab", marker="o", markersize=3)
            ax.fill_between(dates, [row["lower"] for row in item["prediction_intervals"]],
                            [row["upper"] for row in item["prediction_intervals"]], color="#872aab", alpha=.18)
            ax.set_title(item["metric"].title()); ax.set_ylabel(item["metric"].title())
        fig.suptitle("Forecast prediction intervals (95%)"); fig.tight_layout()
        fig.savefig(chart_dir / "forecast_intervals.png"); plt.close(fig)
        made += 1
    return made


def write_forecast_outputs(report: dict, output_dir: str | Path) -> dict:
    """Write machine-readable JSON/CSV and all available forecast charts."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    chart_dir = output / "charts" / "forecast"
    charts = _make_charts(report, chart_dir)
    report["outputs"] = {"json": str(output / "forecast_report.json"),
                         "csv": str(output / "forecast_summary.csv"),
                         "chart_directory": str(chart_dir), "charts_generated": charts}
    safe_report = _safe(report)
    (output / "forecast_report.json").write_text(
        json.dumps(safe_report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "forecast_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "status", "selected_model", "validation_mae", "validation_rmse",
                         "validation_mape_pct", "test_mae", "test_rmse", "test_mape_pct",
                         "forecast_3_month_total", "forecast_6_month_total", "forecast_12_month_total", "trend"])
        for item in report["forecasts"]:
            selected = item.get("selected_model") or {}
            validation, test = item.get("validation_metrics") or {}, item.get("test_metrics") or {}
            horizons = item.get("forecasts_by_horizon") or {}
            writer.writerow([item["metric"], item.get("status"), selected.get("name"), validation.get("mae"),
                             validation.get("rmse"), validation.get("mape_pct"), test.get("mae"), test.get("rmse"),
                             test.get("mape_pct"), (horizons.get("3_months") or {}).get("forecast_total"),
                             (horizons.get("6_months") or {}).get("forecast_total"),
                             (horizons.get("12_months") or {}).get("forecast_total"),
                             (item.get("trend") or {}).get("direction")])
    return report
