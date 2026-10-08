import pandas as pd
import pytest
from src.financial_intelligence.contribution import (
    calculate_contributions, percentage_change, significant_change,
)

def test_percentage_change_and_zero_baseline():
    assert percentage_change(100, 112.4) == pytest.approx(12.4)
    assert percentage_change(100, 90) == pytest.approx(-10)
    assert percentage_change(0, 10) is None

def test_significant_change_threshold_and_edges():
    assert significant_change(100, 106, 5)
    assert not significant_change(100, 104.99, 5)
    assert not significant_change(0, 20, 5)
    with pytest.raises(ValueError): significant_change(10, 20, -1)

def test_contribution_amount_and_share():
    previous = pd.DataFrame({"segment":["A","B"],"sales":[60,40]})
    current = pd.DataFrame({"segment":["A","B"],"sales":[50,30]})
    rows = calculate_contributions(previous,current,dimension="segment",metric="revenue",
                                   value_column="sales",total_change=-20)
    assert rows[0]["absolute_change"] == -10
    assert rows[0]["contribution_to_total_change_pct"] == pytest.approx(50)

def test_contributor_ranking_and_offsets():
    previous = pd.DataFrame({"segment":["A","B"],"sales":[90,10]})
    current = pd.DataFrame({"segment":["A","B"],"sales":[80,30]})
    rows = calculate_contributions(previous,current,dimension="segment",metric="revenue",
                                   value_column="sales",total_change=10)
    assert rows[0]["value"] == "B"
    assert rows[0]["rank"] == 1
    assert rows[0]["contribution_to_total_change_pct"] == pytest.approx(200)
    assert rows[1]["contribution_to_total_change_pct"] == pytest.approx(-100)

def test_order_metric_counts_unique_ids_and_zero_total_delta_edge():
    previous = pd.DataFrame({"segment":["A","A","B"],"order_id":[1,1,2]})
    current = pd.DataFrame({"segment":["A","B","B"],"order_id":[1,3,4]})
    rows = calculate_contributions(previous,current,dimension="segment",metric="orders",
                                   order_id_column="order_id",total_change=1)
    assert {row["value"]: row["absolute_change"] for row in rows} == {"A":0.0,"B":1.0}
    unchanged = calculate_contributions(previous,current,dimension="segment",metric="orders",
                                        order_id_column="order_id",total_change=0)
    assert all(row["contribution_to_total_change_pct"] is None for row in unchanged)
