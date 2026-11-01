from pathlib import Path

import pandas as pd

from src.analytics import execute_query, portfolio_insights


ROOT = Path(__file__).resolve().parents[1]


def test_grounded_default_rate_query():
    account = pd.read_csv(ROOT / "data" / "processed" / "account_scored.csv")
    result = execute_query(account, "default_rate", "loan_type", [], 5)
    expected = account.groupby("loan_type").target_default.mean().max()
    assert abs(result.table.iloc[0]["value"] - expected) < 1e-12
    assert "account_scored.csv" in result.provenance


def test_grounded_lowest_default_rate_query():
    account = pd.read_csv(ROOT / "data" / "processed" / "account_scored.csv")
    result = execute_query(
        account,
        "default_rate",
        "loan_type",
        [],
        5,
        ranking_intent="lowest",
    )
    expected = account.groupby("loan_type").target_default.mean().min()

    assert abs(result.table.iloc[0]["value"] - expected) < 1e-12
    assert result.table.iloc[0]["loan_type"] == "Auto Loan"
    assert "lowest default rate" in result.answer
    assert "ranking: lowest" in result.provenance


def test_query_rejects_unknown_ranking_intent():
    account = pd.read_csv(ROOT / "data" / "processed" / "account_scored.csv")
    try:
        execute_query(
            account,
            "default_rate",
            "loan_type",
            [],
            5,
            ranking_intent="smallest-ish",
        )
    except ValueError as exc:
        assert "ranking intent" in str(exc)
    else:
        raise AssertionError("Unknown ranking intent should be rejected")


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


def test_portfolio_insights_are_honest_for_narrow_filters():
    account = pd.read_csv(ROOT / "data" / "processed" / "account_scored.csv")
    narrow = account.loc[
        account["region"].eq("South")
        & account["employment_type"].eq("Unemployed")
        & account["risk_band"].eq("High")
    ]

    insights = portfolio_insights(narrow)
    rendered = " ".join(finding for finding, _ in insights).casefold()

    assert "nan" not in rendered
    assert "does not meet the 50-account reliability threshold" in rendered
    assert "with at least 50 accounts is" not in rendered

