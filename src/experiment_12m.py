"""Present-state monitoring experiment using all 12 repayment months."""

import json

import src.modeling as modeling
from src.config import MODEL_DIR


FULL_HISTORY_NUMERIC = [
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
    "late_months_12m",
    "missed_months_12m",
    "mean_dpd_12m",
    "max_dpd_12m",
    "mean_payment_ratio_12m",
    "min_payment_ratio_12m",
    "dpd_volatility_12m",
    "max_problem_streak_12m",
    "dpd_deterioration_10m",
    "missed_deterioration_10m",
    "payment_ratio_deterioration_10m",
]


def run_experiment():
    modeling.NUMERIC_FEATURES = FULL_HISTORY_NUMERIC
    modeling.FEATURES = FULL_HISTORY_NUMERIC + modeling.CATEGORICAL_FEATURES
    experiment_dir = MODEL_DIR / "experiment_12m"
    metrics = modeling.train_models(model_dir=experiment_dir)
    metrics["observation_window"] = (
        "All 12 repayment months. This is present-state classification, not an early-warning forecast."
    )
    (experiment_dir / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))
