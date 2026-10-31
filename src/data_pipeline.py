from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, REFERENCE_DATE


def _read_inputs(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    return {
        "customers": pd.read_csv(raw_dir / "customers.csv"),
        "loans": pd.read_csv(raw_dir / "loan_accounts.csv"),
        "bureau": pd.read_csv(raw_dir / "bureau_credit.csv"),
        "repayments": pd.read_csv(raw_dir / "repayment_history.csv"),
        "collections": pd.read_csv(raw_dir / "collections_events.csv"),
    }


def _median_fill_by_group(df: pd.DataFrame, column: str, group: str) -> None:
    missing_flag = f"{column}_missing"
    df[missing_flag] = df[column].isna().astype("int8")
    group_median = df.groupby(group, dropna=False)[column].transform("median")
    df[column] = df[column].fillna(group_median).fillna(df[column].median())


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)


def _consecutive_max(values: pd.Series) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)


def _prepare_customers(customers: pd.DataFrame) -> pd.DataFrame:
    customers = customers.copy()
    customers["gender"] = customers["gender"].replace({"M": "Male", "F": "Female"})
    customers["customer_since"] = pd.to_datetime(customers["customer_since"], errors="coerce")
    for column in ["years_employed", "num_dependents", "years_at_residence"]:
        _median_fill_by_group(customers, column, "employment_type")
    customers["employment_tenure_implausible"] = (
        customers["years_employed"] > customers["age"] - 14
    ).astype("int8")
    customers.loc[customers["employment_tenure_implausible"].eq(1), "years_employed"] = np.nan
    _median_fill_by_group(customers, "years_employed", "employment_type")
    return customers


def _prepare_loans(loans: pd.DataFrame) -> pd.DataFrame:
    loans = loans.copy()
    loans["origination_date"] = pd.to_datetime(loans["origination_date"], errors="coerce")
    _median_fill_by_group(loans, "interest_rate_pct", "loan_type")
    loans["origination_year"] = loans["origination_date"].dt.year.astype("Int64")
    return loans


def _prepare_bureau(bureau: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bureau = bureau.copy()
    reference_date = pd.Timestamp(REFERENCE_DATE)
    bureau["date_opened"] = pd.to_datetime(bureau["date_opened"], errors="coerce")
    bureau["date_closed"] = pd.to_datetime(bureau["date_closed"], errors="coerce")
    bureau["days_past_due_missing"] = bureau["days_past_due"].isna().astype("int8")
    status_median = bureau.groupby("account_status")["days_past_due"].transform("median")
    bureau["days_past_due"] = bureau["days_past_due"].fillna(status_median).fillna(0)
    bureau["future_close_date_flag"] = (bureau["date_closed"] > reference_date).astype("int8")
    bureau["bureau_utilization"] = _safe_ratio(
        bureau["outstanding_amount"], bureau["sanctioned_amount"]
    ).clip(0, 1)
    bureau["line_age_years"] = (
        (reference_date - bureau["date_opened"]).dt.days.div(365.25).clip(lower=0)
    )
    bureau["is_active"] = bureau["account_status"].eq("Active").astype("int8")
    bureau["is_written_off"] = bureau["account_status"].eq("Written-Off").astype("int8")
    bureau["is_delinquent"] = bureau["days_past_due"].gt(0).astype("int8")

    agg = bureau.groupby("customer_id", as_index=False).agg(
        bureau_line_count=("bureau_id", "count"),
        bureau_active_count=("is_active", "sum"),
        bureau_written_off_count=("is_written_off", "sum"),
        bureau_delinquent_count=("is_delinquent", "sum"),
        bureau_total_sanctioned=("sanctioned_amount", "sum"),
        bureau_total_outstanding=("outstanding_amount", "sum"),
        bureau_mean_utilization=("bureau_utilization", "mean"),
        bureau_max_utilization=("bureau_utilization", "max"),
        bureau_mean_dpd=("days_past_due", "mean"),
        bureau_max_dpd=("days_past_due", "max"),
        bureau_oldest_line_years=("line_age_years", "max"),
        bureau_dpd_missing_count=("days_past_due_missing", "sum"),
        bureau_future_close_count=("future_close_date_flag", "sum"),
        bureau_credit_type_diversity=("credit_type", "nunique"),
        bureau_lender_diversity=("lender_category", "nunique"),
    )
    agg["bureau_total_utilization"] = _safe_ratio(
        agg["bureau_total_outstanding"], agg["bureau_total_sanctioned"]
    ).clip(0, 1)
    return bureau, agg


def _repayment_aggregates(repayments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    repayments = repayments.copy().sort_values(["account_id", "payment_month"])
    repayments["is_late"] = repayments["payment_status"].eq("Late").astype("int8")
    repayments["is_missed"] = repayments["payment_status"].eq("Missed").astype("int8")
    repayments["is_on_time"] = repayments["payment_status"].eq("On-Time").astype("int8")
    repayments["payment_ratio"] = _safe_ratio(repayments["emi_paid"], repayments["emi_due"]).clip(0, 1)
    repayments["is_problem"] = repayments["payment_status"].isin(["Late", "Missed"])

    def aggregate_window(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
        grouped = frame.groupby("account_id", as_index=False).agg(
            **{
                f"months_observed_{suffix}": ("payment_month", "nunique"),
                f"on_time_months_{suffix}": ("is_on_time", "sum"),
                f"late_months_{suffix}": ("is_late", "sum"),
                f"missed_months_{suffix}": ("is_missed", "sum"),
                f"mean_dpd_{suffix}": ("days_past_due", "mean"),
                f"max_dpd_{suffix}": ("days_past_due", "max"),
                f"mean_payment_ratio_{suffix}": ("payment_ratio", "mean"),
                f"min_payment_ratio_{suffix}": ("payment_ratio", "min"),
                f"emi_due_total_{suffix}": ("emi_due", "sum"),
                f"emi_paid_total_{suffix}": ("emi_paid", "sum"),
                f"dpd_volatility_{suffix}": ("days_past_due", "std"),
            }
        )
        streak = (
            frame.groupby("account_id")["is_problem"]
            .apply(_consecutive_max)
            .rename(f"max_problem_streak_{suffix}")
            .reset_index()
        )
        return grouped.merge(streak, on="account_id", how="left")

    first_three = aggregate_window(repayments[repayments["payment_month"].le(3)], "3m")
    recent_three = aggregate_window(
        repayments[repayments["payment_month"].between(4, 6)], "recent3m"
    )
    first_five = aggregate_window(repayments[repayments["payment_month"].le(5)], "first5m")
    recent_five = aggregate_window(
        repayments[repayments["payment_month"].between(6, 10)], "recent5m"
    )
    first_six = aggregate_window(repayments[repayments["payment_month"].le(6)], "6m")
    first_ten = aggregate_window(repayments[repayments["payment_month"].le(10)], "10m")
    full = aggregate_window(repayments, "12m")
    latest = (
        repayments.sort_values("payment_month")
        .groupby("account_id", as_index=False)
        .tail(1)[["account_id", "outstanding_balance"]]
        .rename(columns={"outstanding_balance": "latest_outstanding_balance"})
    )
    agg = (
        first_three.merge(recent_three, on="account_id", how="outer")
        .merge(first_five, on="account_id", how="outer")
        .merge(recent_five, on="account_id", how="outer")
        .merge(first_six, on="account_id", how="outer")
        .merge(first_ten, on="account_id", how="outer")
        .merge(full, on="account_id", how="outer")
        .merge(latest, on="account_id", how="left")
    )
    agg["dpd_deterioration_6m"] = agg["mean_dpd_recent3m"] - agg["mean_dpd_3m"]
    agg["missed_deterioration_6m"] = agg["missed_months_recent3m"] - agg["missed_months_3m"]
    agg["payment_ratio_deterioration_6m"] = (
        agg["mean_payment_ratio_3m"] - agg["mean_payment_ratio_recent3m"]
    )
    agg["dpd_deterioration_10m"] = agg["mean_dpd_recent5m"] - agg["mean_dpd_first5m"]
    agg["missed_deterioration_10m"] = (
        agg["missed_months_recent5m"] - agg["missed_months_first5m"]
    )
    agg["payment_ratio_deterioration_10m"] = (
        agg["mean_payment_ratio_first5m"] - agg["mean_payment_ratio_recent5m"]
    )
    agg["balance_reduction_pct"] = 1 - _safe_ratio(
        agg["latest_outstanding_balance"],
        repayments.groupby("account_id")["outstanding_balance"].max().reindex(agg["account_id"]).reset_index(drop=True),
    )
    return repayments, agg


def _prepare_collections(
    collections: pd.DataFrame, loans: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    collections = collections.copy()
    collections["event_date"] = pd.to_datetime(collections["event_date"], errors="coerce")
    collections = collections.merge(
        loans[["account_id", "origination_date"]], on="account_id", how="left", validate="many_to_one"
    )
    collections["pre_origination_flag"] = (
        collections["event_date"] < collections["origination_date"]
    ).astype("int8")
    collections["valid_for_time_analysis"] = collections["pre_origination_flag"].eq(0)
    valid = collections[collections["valid_for_time_analysis"]]
    agg = valid.groupby("account_id", as_index=False).agg(
        valid_collection_events=("collection_event_id", "count"),
        amount_recovered=("amount_recovered", "sum"),
        collection_action_diversity=("action_type", "nunique"),
        collection_last_event=("event_date", "max"),
    )
    invalid_counts = collections.groupby("account_id", as_index=False).agg(
        invalid_collection_events=("pre_origination_flag", "sum")
    )
    return collections, agg.merge(invalid_counts, on="account_id", how="outer")


def build_analysis_data(
    raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR
) -> tuple[pd.DataFrame, dict[str, int | float | str]]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    data = _read_inputs(raw_dir)
    customers = _prepare_customers(data["customers"])
    loans = _prepare_loans(data["loans"])
    bureau, bureau_agg = _prepare_bureau(data["bureau"])
    repayments, repayment_agg = _repayment_aggregates(data["repayments"])
    collections, collection_agg = _prepare_collections(data["collections"], loans)

    account = loans.merge(customers, on="customer_id", how="left", validate="many_to_one")
    account = account.merge(bureau_agg, on="customer_id", how="left", validate="many_to_one")
    account = account.merge(repayment_agg, on="account_id", how="left", validate="one_to_one")
    account = account.merge(collection_agg, on="account_id", how="left", validate="one_to_one")

    bureau_columns = [column for column in account if column.startswith("bureau_")]
    account[bureau_columns] = account[bureau_columns].fillna(0)
    collection_columns = [
        "valid_collection_events",
        "amount_recovered",
        "collection_action_diversity",
        "invalid_collection_events",
    ]
    account[collection_columns] = account[collection_columns].fillna(0)
    account["customer_since_after_origination"] = (
        account["customer_since"] > account["origination_date"]
    ).astype("int8")
    account["customer_loan_count"] = account.groupby("customer_id")["account_id"].transform("count")
    account["loan_to_income"] = _safe_ratio(account["loan_amount"], account["annual_income"])
    account["bureau_outstanding_to_income"] = _safe_ratio(
        account["bureau_total_outstanding"], account["annual_income"]
    )
    account["age_band"] = pd.cut(
        account["age"], [0, 25, 35, 45, 55, 65, 200], right=False,
        labels=["Under 25", "25–34", "35–44", "45–54", "55–64", "65+"],
    ).astype("string")
    account["income_band"] = pd.qcut(
        account["annual_income"], 5,
        labels=["Q1 Lowest", "Q2", "Q3", "Q4", "Q5 Highest"],
        duplicates="drop",
    ).astype("string")
    account["exposure_at_risk"] = account["latest_outstanding_balance"] * account["target_default"]

    if len(account) != len(loans) or account["account_id"].duplicated().any():
        raise ValueError("Account-grain contract failed: expected exactly one row per loan account.")

    quality = {
        "generated_at_reference_date": REFERENCE_DATE,
        "account_rows": int(len(account)),
        "customer_rows": int(len(customers)),
        "default_count": int(account["target_default"].sum()),
        "default_rate": float(account["target_default"].mean()),
        "gender_labels_standardized": int(data["customers"]["gender"].isin(["M", "F"]).sum()),
        "employment_tenure_implausible": int(customers["employment_tenure_implausible"].sum()),
        "customer_since_after_origination": int(account["customer_since_after_origination"].sum()),
        "collections_before_origination": int(collections["pre_origination_flag"].sum()),
        "bureau_future_close_dates": int(bureau["future_close_date_flag"].sum()),
        "bureau_customers_missing": int(account.loc[account["bureau_line_count"].eq(0), "customer_id"].nunique()),
    }

    account.to_csv(processed_dir / "account_master.csv", index=False)
    repayments.to_csv(processed_dir / "repayment_clean.csv", index=False)
    bureau.to_csv(processed_dir / "bureau_clean.csv", index=False)
    collections.to_csv(processed_dir / "collections_review.csv", index=False)
    pd.DataFrame([quality]).to_csv(processed_dir / "data_quality_summary.csv", index=False)
    (processed_dir / "data_quality_summary.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )
    return account, quality


if __name__ == "__main__":
    frame, summary = build_analysis_data()
    print(f"Built {len(frame):,} account-grain rows")
    print(json.dumps(summary, indent=2))
