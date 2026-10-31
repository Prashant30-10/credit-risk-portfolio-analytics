"""Leakage-safe comparison using repayment months 1–10 as features."""

import json

import src.modeling as modeling
from src.config import MODEL_DIR


TEN_MONTH_NUMERIC = [
    "age",
    "annual_income",
    "years_employed",
    "num_dependents",
    "years_at_residence",
    "loan_amount",
    "interest_rate_pct",
    "tenure_months",
    "collateral_flag",
    "customer_loan_count",
    "loan_to_income",
    "bureau_outstanding_to_income",
    "bureau_line_count",
    "bureau_active_count",
    "bureau_written_off_count",
    "bureau_delinquent_count",
    "bureau_total_sanctioned",
    "bureau_total_outstanding",
    "bureau_total_utilization",
    "bureau_mean_dpd",
    "bureau_max_dpd",
    "bureau_oldest_line_years",
    "bureau_credit_type_diversity",
    "bureau_lender_diversity",
    "late_months_10m",
    "missed_months_10m",
    "mean_dpd_10m",
    "max_dpd_10m",
    "mean_payment_ratio_10m",
    "min_payment_ratio_10m",
    "dpd_volatility_10m",
    "max_problem_streak_10m",
    "dpd_deterioration_10m",
    "missed_deterioration_10m",
    "payment_ratio_deterioration_10m",
]


def run_experiment():
    modeling.NUMERIC_FEATURES = TEN_MONTH_NUMERIC
    modeling.FEATURES = TEN_MONTH_NUMERIC + modeling.CATEGORICAL_FEATURES
    experiment_dir = MODEL_DIR / "experiment_10m"
    metrics = modeling.train_models(model_dir=experiment_dir)
    metrics["observation_window"] = (
        "Repayment months 1–10 only; months 11–12 excluded. "
        "Validation and test remain customer-separated account holdouts because target_default is account-level."
    )
    (experiment_dir / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))
