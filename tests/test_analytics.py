from pathlib import Path

import pandas as pd

from src.analytics import execute_query


ROOT = Path(__file__).resolve().parents[1]


def test_grounded_default_rate_query():
    account = pd.read_csv(ROOT / "data" / "processed" / "account_scored.csv")
    result = execute_query(account, "default_rate", "loan_type", [], 5)
    expected = account.groupby("loan_type").target_default.mean().max()
    assert abs(result.table.iloc[0]["value"] - expected) < 1e-12
    assert "account_scored.csv" in result.provenance


def test_query_rejects_unknown_filter_value():
    account = pd.read_csv(ROOT / "data" / "processed" / "account_scored.csv")
    try:
        execute_query(
            account,
            "default_rate",
            "loan_type",
            [{"field": "region", "value": "Atlantis"}],
            5,
        )
    except ValueError as exc:
        assert "No value" in str(exc)
    else:
        raise AssertionError("Unknown filter value should be rejected")

