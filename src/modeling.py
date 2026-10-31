from __future__ import annotations

import json
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from lightgbm import LGBMClassifier

from src.config import MODEL_DIR, PROCESSED_DIR, RANDOM_STATE


NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "gender",
    "marital_status",
    "education_level",
    "employment_type",
    "occupation_sector",
    "region",
    "city_tier",
    "housing_type",
    "loan_type",
    "loan_purpose",
    "disbursal_channel",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _ks_statistic(y_true: pd.Series | np.ndarray, probability: np.ndarray) -> float:
    false_positive, true_positive, _ = roc_curve(y_true, probability)
    return float(np.max(true_positive - false_positive))


def _metrics(y_true: pd.Series, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction).ravel()
    auc = roc_auc_score(y_true, probability)
    return {
        "roc_auc": float(auc),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "ks": _ks_statistic(y_true, probability),
        "gini": float(2 * auc - 1),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "threshold": float(threshold),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def _choose_threshold(y_true: pd.Series, probability: np.ndarray, recall_floor: float = 0.70) -> float:
    candidates = []
    for threshold in np.linspace(0.05, 0.90, 171):
        prediction = probability >= threshold
        recall = recall_score(y_true, prediction, zero_division=0)
        precision = precision_score(y_true, prediction, zero_division=0)
        f1 = f1_score(y_true, prediction, zero_division=0)
        candidates.append((threshold, recall, precision, f1))
    eligible = [row for row in candidates if row[1] >= recall_floor]
    pool = eligible or candidates
    return float(max(pool, key=lambda row: (row[3], row[2]))[0])


def _split_by_customer(frame: pd.DataFrame):
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    development_index, test_index = next(
        outer.split(frame, frame["target_default"], groups=frame["customer_id"])
    )
    development = frame.iloc[development_index].copy()
    test = frame.iloc[test_index].copy()
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE + 1)
    train_index, validation_index = next(
        inner.split(
            development,
            development["target_default"],
            groups=development["customer_id"],
        )
    )
    return development.iloc[train_index], development.iloc[validation_index], test


def _logistic_pipeline() -> Pipeline:
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]
    )
    preprocess = ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                    C=0.5,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _boosting_pipeline() -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    preprocess = ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )
    categorical_indices = list(range(len(NUMERIC_FEATURES), len(FEATURES)))
    model = HistGradientBoostingClassifier(
        learning_rate=0.055,
        max_iter=250,
        max_leaf_nodes=24,
        min_samples_leaf=30,
        l2_regularization=1.0,
        class_weight="balanced",
        categorical_features=categorical_indices,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def _lightgbm_pipeline() -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]
    )
    preprocess = ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )
    model = LGBMClassifier(
        objective="binary",
        n_estimators=450,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=5,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=-1,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def _risk_indicator(row: pd.Series) -> str:
    indicators: list[tuple[float, str]] = []
    indicators.append((float(row.get("max_dpd_10m", 0)) / 90, "elevated observation-period days past due"))
    indicators.append((float(row.get("missed_months_10m", 0)) / 10, "missed payments in the first ten months"))
    indicators.append((float(row.get("bureau_written_off_count", 0)) / 3, "external written-off credit lines"))
    indicators.append((float(row.get("bureau_total_utilization", 0)), "high external credit utilization"))
    income_pressure = max(0.0, (80000 - float(row.get("annual_income", 80000))) / 80000)
    indicators.append((income_pressure, "lower income capacity"))
    selected = [text for score, text in sorted(indicators, reverse=True) if score > 0.15][:3]
    return "; ".join(selected) if selected else "no dominant adverse indicator"


def train_models(
    account_path=PROCESSED_DIR / "account_master.csv",
    model_dir=MODEL_DIR,
):
    model_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(account_path)
    train, validation, test = _split_by_customer(frame)
    X_train, y_train = train[FEATURES], train["target_default"]
    X_validation, y_validation = validation[FEATURES], validation["target_default"]
    X_test, y_test = test[FEATURES], test["target_default"]

    candidates = {
        "logistic_regression": _logistic_pipeline(),
        "hist_gradient_boosting": _boosting_pipeline(),
        "lightgbm": _lightgbm_pipeline(),
    }
    validation_metrics: dict[str, dict[str, float | int]] = {}
    thresholds: dict[str, float] = {}
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        probability = pipeline.predict_proba(X_validation)[:, 1]
        threshold = _choose_threshold(y_validation, probability)
        thresholds[name] = threshold
        validation_metrics[name] = _metrics(y_validation, probability, threshold)

    selected_name = max(
        candidates,
        key=lambda name: (
            validation_metrics[name]["f1"],
            validation_metrics[name]["pr_auc"],
        ),
    )
    selected = candidates[selected_name]
    threshold = thresholds[selected_name]
    test_probability = selected.predict_proba(X_test)[:, 1]
    test_metrics = _metrics(y_test, test_probability, threshold)
    test_base_rate = float(y_test.mean())
    top_twenty_cutoff = float(np.quantile(test_probability, 0.80))
    top_twenty_prediction = test_probability >= top_twenty_cutoff
    test_metrics["precision_at_top_20_pct"] = float(
        precision_score(y_test, top_twenty_prediction, zero_division=0)
    )
    test_metrics["recall_at_top_20_pct"] = float(
        recall_score(y_test, top_twenty_prediction, zero_division=0)
    )
    test_metrics["precision_lift_at_top_20_pct"] = float(
        test_metrics["precision_at_top_20_pct"] / test_base_rate
    )
    test_metrics["pr_auc_lift_vs_base_rate"] = float(test_metrics["pr_auc"] / test_base_rate)

    importance = permutation_importance(
        selected,
        X_validation,
        y_validation,
        scoring="average_precision",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    all_probability = selected.predict_proba(frame[FEATURES])[:, 1]
    frame["risk_score"] = (all_probability * 100).round(2)
    high_band_cutoff = float(np.quantile(all_probability, 0.80))
    medium_band_cutoff = float(np.quantile(all_probability, 0.50))
    frame["risk_band"] = np.select(
        [all_probability >= high_band_cutoff, all_probability >= medium_band_cutoff],
        ["High", "Medium"],
        default="Low",
    )
    frame["key_risk_indicators"] = frame.apply(_risk_indicator, axis=1)
    frame.to_csv(PROCESSED_DIR / "account_scored.csv", index=False)
    importance_frame.to_csv(model_dir / "feature_importance.csv", index=False)

    segment_rows = []
    test_scored = test.copy()
    test_scored["probability"] = test_probability
    for dimension in ["loan_type", "employment_type", "region"]:
        for value, group in test_scored.groupby(dimension):
            if group["target_default"].nunique() < 2 or len(group) < 50:
                continue
            segment_rows.append(
                {
                    "dimension": dimension,
                    "segment": value,
                    "accounts": len(group),
                    "default_rate": group["target_default"].mean(),
                    "roc_auc": roc_auc_score(group["target_default"], group["probability"]),
                    "ks": _ks_statistic(group["target_default"], group["probability"].to_numpy()),
                }
            )
    pd.DataFrame(segment_rows).to_csv(model_dir / "segment_metrics.csv", index=False)

    metrics = {
        "selected_model": selected_name,
        "selection_rule": "Highest validation F1 with a 70% recall floor; PR-AUC breaks ties.",
        "observation_window": "Repayment months 1–10 only; months 11–12 excluded from model features.",
        "split_strategy": "Customer-grouped 60/20/20 train/validation/test split.",
        "train_accounts": int(len(train)),
        "validation_accounts": int(len(validation)),
        "test_accounts": int(len(test)),
        "validation": validation_metrics,
        "test": test_metrics,
    }
    (model_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "model": selected,
            "features": FEATURES,
            "threshold": threshold,
            "high_band_cutoff": high_band_cutoff,
            "medium_band_cutoff": medium_band_cutoff,
            "selected_name": selected_name,
        },
        model_dir / "credit_risk_model.joblib",
    )
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_models(), indent=2))

