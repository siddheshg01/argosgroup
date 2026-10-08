"""Period-over-period deltas and dimension-level contribution analysis."""
from __future__ import annotations

import math
import pandas as pd


def percentage_change(previous_value: float, current_value: float) -> float | None:
    """Return percent change; return None when a zero baseline makes it undefined."""
    previous_value, current_value = float(previous_value), float(current_value)
    if previous_value == 0 or not math.isfinite(previous_value) or not math.isfinite(current_value):
        return None
    return (current_value - previous_value) / previous_value * 100


def significant_change(previous_value: float, current_value: float, threshold_pct: float = 5.0) -> bool:
    """A change is significant when its absolute percentage reaches the threshold."""
    if threshold_pct < 0:
        raise ValueError("threshold_pct must be non-negative")
    change = percentage_change(previous_value, current_value)
    return change is not None and abs(change) >= threshold_pct


def _aggregate(frame: pd.DataFrame, dimension: str, metric: str,
               value_column: str | None, order_id_column: str | None) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="float64")
    if metric == "orders":
        if not order_id_column or order_id_column not in frame:
            return pd.Series(dtype="float64")
        grouped = frame.groupby(dimension, dropna=False)[order_id_column].nunique(dropna=True)
    else:
        if not value_column or value_column not in frame:
            return pd.Series(dtype="float64")
        values = pd.to_numeric(frame[value_column], errors="coerce")
        grouped = values.groupby(frame[dimension], dropna=False).sum(min_count=1)
    return grouped.astype(float)


def calculate_contributions(previous: pd.DataFrame, current: pd.DataFrame, *,
                            dimension: str, metric: str, total_change: float,
                            value_column: str | None = None,
                            order_id_column: str | None = None,
                            labels: dict | None = None, top_n: int = 10) -> list[dict]:
    """Compare dimension values and rank their signed contribution to total change.

    A positive contribution means the segment moved in the same direction as
    the total metric change. Negative contributions offset the total movement.
    Shares can exceed 100% when other segments moved in the opposite direction.
    """
    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    before = _aggregate(previous, dimension, metric, value_column, order_id_column).rename("previous_value")
    after = _aggregate(current, dimension, metric, value_column, order_id_column).rename("current_value")
    joined = pd.concat([before, after], axis=1).fillna(0)
    joined["absolute_change"] = joined["current_value"] - joined["previous_value"]
    if total_change:
        joined["contribution_pct"] = joined["absolute_change"] / total_change * 100
    else:
        joined["contribution_pct"] = float("nan")
    joined["percentage_change"] = [percentage_change(row.previous_value, row.current_value)
                                   for row in joined.itertuples()]
    joined = joined.sort_values(["contribution_pct", "absolute_change"], ascending=False,
                                na_position="last").head(top_n)
    result = []
    for rank, (raw_label, row) in enumerate(joined.iterrows(), 1):
        value = labels.get(str(raw_label), str(raw_label)) if labels else str(raw_label)
        if pd.isna(raw_label):
            value = "Unknown"
        result.append({
            "rank": rank,
            "dimension": dimension,
            "value": value,
            "previous_value": float(row["previous_value"]),
            "current_value": float(row["current_value"]),
            "absolute_change": float(row["absolute_change"]),
            "percentage_change": None if pd.isna(row["percentage_change"]) else float(row["percentage_change"]),
            "contribution_to_total_change_pct": None if pd.isna(row["contribution_pct"]) else float(row["contribution_pct"]),
        })
    return result
