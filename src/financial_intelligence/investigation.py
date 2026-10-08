"""Reusable Phase 2 month-over-month root-cause investigation engine."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .loader import load_dataset
from .schema import discover_schema
from .cleaner import clean_dataset
from .features import engineer_features
from .contribution import calculate_contributions, percentage_change, significant_change


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (float, int)):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


class RootCauseInvestigator:
    """Find significant changes and rank associated segment contributions.

    Uses Phase 1 loading, schema discovery, cleaning, and feature engineering.
    The default significance rule is an absolute month-over-month change of
    at least 5%. Zero baselines have undefined percentages and are not selected.
    """
    def __init__(self, dataset_path="data/raw/amazon_sales_2024.csv", output_dir="output",
                 threshold_pct: float = 5.0, top_n: int = 10,
                 max_chart_investigations: int = 12):
        if threshold_pct < 0:
            raise ValueError("threshold_pct must be non-negative")
        if top_n < 0 or max_chart_investigations < 0:
            raise ValueError("top_n and max_chart_investigations must be non-negative")
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.threshold_pct = float(threshold_pct)
        self.top_n = int(top_n)
        self.max_chart_investigations = int(max_chart_investigations)

    @staticmethod
    def _total(frame: pd.DataFrame, metric: str, mapping: dict) -> float:
        if metric == "orders":
            return float(frame[mapping["order_id"]].nunique(dropna=True))
        return float(pd.to_numeric(frame[mapping[metric]], errors="coerce").sum())

    def _investigate_pair(self, previous: pd.DataFrame, current: pd.DataFrame,
                          previous_period: str, current_period: str,
                          metric: str, mapping: dict, customer_labels: dict) -> dict:
        prev_value = self._total(previous, metric, mapping)
        curr_value = self._total(current, metric, mapping)
        delta = curr_value - prev_value
        pct = percentage_change(prev_value, curr_value)
        dimensions = {
            "location": mapping.get("location"),
            "region": mapping.get("region"),
            "category": mapping.get("category"),
            "product": mapping.get("product"),
            "customer": mapping.get("customer_id"),
        }
        contributors = {}
        for name, column in dimensions.items():
            if column and column in previous.columns and column in current.columns:
                labels = customer_labels if name == "customer" else None
                contributors[name] = calculate_contributions(
                    previous, current, dimension=column, metric=metric, total_change=delta,
                    value_column=mapping.get(metric), order_id_column=mapping.get("order_id"),
                    labels=labels, top_n=self.top_n)
        direction = "increased" if delta > 0 else "decreased" if delta < 0 else "was unchanged"
        evidence = [f"{metric.title()} {direction} {abs(pct):.2f}% from {previous_period} to {current_period}."] if pct is not None else []
        verb = "increase" if delta > 0 else "decline"
        for dimension in ("product", "category", "location", "region", "customer"):
            relevant = [row for row in contributors.get(dimension, [])
                        if row["contribution_to_total_change_pct"] is not None
                        and row["contribution_to_total_change_pct"] > 0]
            if relevant:
                first = relevant[0]
                evidence.append(f"{first['value']} ({dimension}) contributed to {first['contribution_to_total_change_pct']:.2f}% of the {metric} {verb} ({first['absolute_change']:+,.2f}) between {previous_period} and {current_period}.")
        return {
            "metric": metric,
            "period": current_period,
            "previous_period": previous_period,
            "current_value": curr_value,
            "previous_value": prev_value,
            "absolute_change": delta,
            "percentage_change": pct,
            "contributors": contributors,
            "evidence": evidence,
        }

    def _charts(self, investigations: list[dict], chart_dir: Path) -> int:
        chart_dir.mkdir(parents=True, exist_ok=True)
        ranked = sorted(investigations, key=lambda item: abs(item["percentage_change"] or 0), reverse=True)
        made = 0
        for item in ranked[:self.max_chart_investigations]:
            for dimension in ("product", "category", "location"):
                contributors = item["contributors"].get(dimension, [])
                values = [row for row in contributors if row["contribution_to_total_change_pct"] is not None][:8]
                if not values:
                    continue
                labels = [row["value"] for row in values][::-1]
                shares = [row["contribution_to_total_change_pct"] for row in values][::-1]
                fig, ax = plt.subplots(figsize=(9, max(3, 0.42 * len(values))))
                colors = ["#a23bc4" if value >= 0 else "#9aa0a6" for value in shares]
                ax.barh(labels, shares, color=colors)
                ax.axvline(0, color="#666", linewidth=.8)
                ax.set(title=f"{item['metric'].title()} contributors: {dimension.title()} ({item['period']})",
                       xlabel="Contribution to total change (%)", ylabel=dimension.title())
                fig.tight_layout()
                safe_period = re.sub(r"[^A-Za-z0-9_-]", "_", item["period"])
                fig.savefig(chart_dir / f"{item['metric']}_{safe_period}_{dimension}.png")
                plt.close(fig)
                made += 1
        return made

    def run(self) -> dict:
        """Load and prepare source data, investigate adjacent months, and write outputs."""
        raw = load_dataset(self.dataset_path)
        schema = discover_schema(raw)
        cleaned, clean_summary = clean_dataset(raw, schema)
        featured, feature_info = engineer_features(cleaned, schema)
        mapping = schema["mapping"]
        if "date" not in mapping:
            raise ValueError("Root cause investigation requires a recognized date column.")
        featured = featured.copy()
        featured["_phase2_period"] = pd.to_datetime(featured[mapping["date"]], errors="coerce").dt.to_period("M").astype("string")
        periods = sorted(featured["_phase2_period"].dropna().unique())
        metrics = [metric for metric in ("revenue", "quantity", "orders")
                   if (metric == "orders" and "order_id" in mapping)
                   or (metric == "quantity" and "quantity" in mapping)
                   or (metric == "revenue" and "revenue" in mapping)]
        customer_col = mapping.get("customer_id")
        customer_labels = {}
        if customer_col:
            ids = sorted(featured[customer_col].dropna().astype(str).unique())
            customer_labels = {value: f"Customer {index:05d}" for index, value in enumerate(ids, 1)}
        investigations = []
        for previous_period, current_period in zip(periods, periods[1:]):
            prev_month, current_month = pd.Period(previous_period), pd.Period(current_period)
            if current_month != prev_month + 1:
                continue  # don't label gaps in source coverage as month-over-month
            previous = featured.loc[featured["_phase2_period"] == previous_period]
            current = featured.loc[featured["_phase2_period"] == current_period]
            for metric in metrics:
                previous_value = self._total(previous, metric, mapping)
                current_value = self._total(current, metric, mapping)
                if significant_change(previous_value, current_value, self.threshold_pct):
                    investigations.append(self._investigate_pair(previous, current, previous_period,
                                                                 current_period, metric, mapping, customer_labels))

        report = {
            "metadata": {
                "dataset": self.dataset_path.name,
                "phase": "Phase 2 - Root Cause Investigation Engine",
                "records_analyzed": int(len(featured)),
                "significance_threshold_pct": self.threshold_pct,
                "significance_rule": "absolute month-over-month percentage change >= configured threshold; zero baseline percentage is undefined and excluded",
                "feature_formulas": feature_info.get("formulas", {}),
                "cleaning_summary": {key: value for key, value in clean_summary.items() if key != "quarantined_records"},
            },
            "metrics_investigated": metrics,
            "significant_changes": investigations,
            "significant_change_count": len(investigations),
            "root_cause_investigation_count": len(investigations),
            "interpretation": "Contributions describe segment changes associated with the overall metric change; they do not establish causation.",
        }
        report = _json_safe(report)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "root_cause_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        csv_path = self.output_dir / "root_cause_summary.csv"
        import csv
        with csv_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["metric", "previous_period", "period", "previous_value", "current_value",
                             "absolute_change", "percentage_change", "top_product", "top_category",
                             "top_location", "evidence"])
            for item in investigations:
                def top(name):
                    rows = item["contributors"].get(name, [])
                    return rows[0]["value"] if rows else ""
                writer.writerow([item["metric"], item["previous_period"], item["period"],
                                 item["previous_value"], item["current_value"], item["absolute_change"],
                                 item["percentage_change"], top("product"), top("category"), top("location"),
                                 " | ".join(item["evidence"])])
        charts_generated = self._charts(investigations, self.output_dir / "charts" / "root_cause")
        report["outputs"] = {"json": "output/root_cause_report.json",
                             "csv": "output/root_cause_summary.csv",
                             "chart_directory": "output/charts/root_cause",
                             "charts_generated": charts_generated}
        # Rewrite after adding the artifact count to metadata.
        (self.output_dir / "root_cause_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        return report
