"""Target-independent data-quality and outlier-filter experiments.

The customer split is created before exclusions. Hard exclusions use only a
known invalid-data flag. Isolation Forest is fitted on training features only;
the resulting training-score cutoff is then applied unchanged to validation
and test accounts.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

import src.modeling as modeling
from src.config import MODEL_DIR, PROCESSED_DIR, RANDOM_STATE


SCENARIOS = {
    "hard_dq_only": 0.0,
    "hard_dq_plus_2_5pct_outliers": 0.025,
    "hard_dq_plus_5pct_outliers": 0.05,
}


def _prepare_outlier_scores(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame):
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    train_values = scaler.fit_transform(imputer.fit_transform(train[modeling.NUMERIC_FEATURES]))
    validation_values = scaler.transform(imputer.transform(validation[modeling.NUMERIC_FEATURES]))
    test_values = scaler.transform(imputer.transform(test[modeling.NUMERIC_FEATURES]))
    detector = IsolationForest(
        n_estimators=400,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    detector.fit(train_values)
    return (
        detector.score_samples(train_values),
        detector.score_samples(validation_values),
        detector.score_samples(test_values),
    )


def _filter(frame: pd.DataFrame, score: np.ndarray, cutoff: float | None) -> pd.DataFrame:
    if cutoff is None:
        return frame.copy()
    return frame.loc[score > cutoff].copy()


def run_experiments():
    frame = pd.read_csv(PROCESSED_DIR / "account_master.csv")
    train_all, validation_all, test_all = modeling._split_by_customer(frame)

    # Employment tenure that exceeds age minus 14 is a genuine invalid row-level
    # condition. Other bad dates are excluded from their feature calculations,
    # not used as reasons to discard an otherwise valid loan account.
    train = train_all.loc[train_all["employment_tenure_implausible"].eq(0)].copy()
    validation = validation_all.loc[validation_all["employment_tenure_implausible"].eq(0)].copy()
    test = test_all.loc[test_all["employment_tenure_implausible"].eq(0)].copy()

    train_score, validation_score, test_score = _prepare_outlier_scores(train, validation, test)
    validation_results = []
    fitted = {}

    for scenario, outlier_fraction in SCENARIOS.items():
        cutoff = None if outlier_fraction == 0 else float(np.quantile(train_score, outlier_fraction))
        train_filtered = _filter(train, train_score, cutoff)
        validation_filtered = _filter(validation, validation_score, cutoff)
        test_filtered = _filter(test, test_score, cutoff)

        candidates = {
            "logistic_regression": modeling._logistic_pipeline(),
            "hist_gradient_boosting": modeling._boosting_pipeline(),
            "lightgbm": modeling._lightgbm_pipeline(),
        }
        for name, pipeline in candidates.items():
            pipeline.fit(
                train_filtered[modeling.FEATURES], train_filtered["target_default"]
            )
            probability = pipeline.predict_proba(validation_filtered[modeling.FEATURES])[:, 1]
            threshold = modeling._choose_threshold(
                validation_filtered["target_default"], probability
            )
            metrics = modeling._metrics(
                validation_filtered["target_default"], probability, threshold
            )
            row = {
                "scenario": scenario,
                "model": name,
                "outlier_fraction_after_hard_dq": outlier_fraction,
                "train_rows": len(train_filtered),
                "validation_rows": len(validation_filtered),
                "test_rows_after_rule": len(test_filtered),
                "total_validation_exclusion_rate": 1 - len(validation_filtered) / len(validation_all),
                **metrics,
            }
            validation_results.append(row)
            fitted[(scenario, name)] = (
                pipeline,
                threshold,
                test_filtered,
                cutoff,
            )

    # The choice is made without test metrics. Prioritize validation KS above
    # 0.30, then F1; otherwise choose highest KS with recall at least 0.70.
    eligible = [row for row in validation_results if row["recall"] >= 0.70]
    above_target = [row for row in eligible if row["ks"] >= 0.30]
    selection_pool = above_target or eligible or validation_results
    best = max(selection_pool, key=lambda row: (row["ks"], row["f1"]))
    pipeline, threshold, test_filtered, cutoff = fitted[(best["scenario"], best["model"])]
    test_probability = pipeline.predict_proba(test_filtered[modeling.FEATURES])[:, 1]
    test_metrics = modeling._metrics(
        test_filtered["target_default"], test_probability, threshold
    )

    output = {
        "selection_policy": (
            "Select on validation only: KS >= 0.30 and recall >= 0.70, then F1; "
            "otherwise highest validation KS subject to recall >= 0.70."
        ),
        "hard_exclusion": "employment_tenure_implausible == 1",
        "raw_bureau_handling": (
            "No account is dropped for a bureau date issue; bureau-derived date quality "
            "flags remain available for audit."
        ),
        "original_split_rows": {
            "train": len(train_all),
            "validation": len(validation_all),
            "test": len(test_all),
        },
        "validation_candidates": validation_results,
        "selected": {
            "scenario": best["scenario"],
            "model": best["model"],
            "threshold": threshold,
            "outlier_score_cutoff": cutoff,
            "test_rows": len(test_filtered),
            "test_exclusion_rate": 1 - len(test_filtered) / len(test_all),
            "validation_metrics": {
                key: best[key]
                for key in ["roc_auc", "pr_auc", "ks", "gini", "precision", "recall", "f1"]
            },
            "test_metrics": test_metrics,
        },
    }
    destination = MODEL_DIR / "experiment_outlier_filter.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    pd.DataFrame(validation_results).to_csv(
        MODEL_DIR / "experiment_outlier_validation.csv", index=False
    )
    return output


if __name__ == "__main__":
    print(json.dumps(run_experiments(), indent=2))
