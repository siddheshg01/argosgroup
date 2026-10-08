"""Orchestrate deterministic Phase 1 analysis and write reports/charts."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from .loader import load_dataset
from .schema import discover_schema
from .validator import validate_dataset
from .cleaner import clean_dataset
from .features import engineer_features
from .kpis import calculate_kpis
from .segmentation import analyze_segments, concentration_analysis
from .trends import analyze_trends
from .risk import calculate_risk
from .report import build_report, save_report

class FinancialIntelligencePipeline:
    """Run the complete dataset-driven analysis using configurable paths."""
    def __init__(self, dataset_path="data/raw/amazon_sales_2024.csv", output_dir="output"):
        self.dataset_path, self.output_dir = Path(dataset_path), Path(output_dir)

    def _charts(self, frame, schema, trends):
        charts = self.output_dir / "charts"; charts.mkdir(parents=True, exist_ok=True)
        m = schema["mapping"]
        if trends and trends.get("monthly"):
            df = pd.DataFrame(trends["monthly"])
            for metric, title in (("revenue", "Monthly Revenue"), ("profit", "Monthly Profit")):
                if metric in df:
                    view = df.dropna(subset=[metric]);
                    if len(view):
                        fig, ax = plt.subplots(figsize=(9,4)); ax.plot(view["_month"], view[metric], marker="o"); ax.set(title=title, xlabel="Month", ylabel=metric.title()); ax.tick_params(axis="x", rotation=45); fig.tight_layout(); fig.savefig(charts / f"monthly_{metric}.png"); plt.close(fig)
        revenue = m.get("revenue")
        for field, filename, title in (("product","revenue_by_product.png","Revenue by Product"),("location","revenue_by_location.png","Revenue by Location")):
            col=m.get(field)
            if revenue and col:
                vals=frame.groupby(col)[revenue].sum().nlargest(10).sort_values()
                if len(vals):
                    fig,ax=plt.subplots(figsize=(9,5)); vals.plot(kind="barh",ax=ax); ax.set(title=title,xlabel="Revenue",ylabel=field.title()); fig.tight_layout(); fig.savefig(charts/filename); plt.close(fig)
        if revenue and m.get("profit") and m.get("product"):
            work = frame.copy()
            work["_profit"] = pd.to_numeric(work[m["profit"]], errors="coerce")
            work["_revenue"] = pd.to_numeric(work[revenue], errors="coerce")
            sums = work.groupby(m["product"])[["_profit", "_revenue"]].sum()
            margins = (sums["_profit"].div(sums["_revenue"].replace(0, pd.NA)) * 100).dropna().nlargest(10).sort_values()
            if len(margins):
                fig, ax = plt.subplots(figsize=(9, 5)); margins.plot(kind="barh", ax=ax)
                ax.set(title="Profit Margin by Product", xlabel="Profit margin (%)", ylabel="Product")
                fig.tight_layout(); fig.savefig(charts / "profit_margin_by_product.png"); plt.close(fig)

    def run(self):
        """Run inspection, validation, conservative cleaning, analysis, and outputs."""
        raw = load_dataset(self.dataset_path)
        schema = discover_schema(raw)
        quality = validate_dataset(raw, schema)
        cleaned, cleaning = clean_dataset(raw, schema)
        featured, feature_info = engineer_features(cleaned, schema)
        kpis = calculate_kpis(featured, schema)
        segments = analyze_segments(featured, schema)
        trends = analyze_trends(featured, schema)
        concentration = concentration_analysis(featured, schema)
        risk = calculate_risk(trends, concentration, quality)
        report = build_report(schema, quality, cleaning, kpis, segments, trends, concentration, risk,
                              len(featured), feature_info.get("formulas"), raw.attrs.get("inspection"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_report(report, self.output_dir / "financial_report.json")
        with (self.output_dir / "financial_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer=csv.writer(f); writer.writerow(["Metric","Value"]); writer.writerows(kpis.items())
        self._charts(featured, schema, trends)
        return report
