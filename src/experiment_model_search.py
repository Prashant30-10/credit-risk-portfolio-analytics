"""Validation-only model search for stronger, defensible rank separation.

The search keeps the original customer-grouped train/validation/test split.
Candidate choice and blend weights use validation data only. The untouched test
partition is scored exactly once after a winner has been selected.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, OrdinalEncoder, StandardScaler

import src.modeling as modeling
from src.config import MODEL_DIR, PROCESSED_DIR, RANDOM_STATE


def _categorical_one_hot():
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]
    )


def _linear_preprocess() -> ColumnTransformer:
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, modeling.NUMERIC_FEATURES),
            ("categorical", _categorical_one_hot(), modeling.CATEGORICAL_FEATURES),
        ]
    )


def _scorecard_preprocess(n_bins: int, include_linear: bool) -> ColumnTransformer:
    transformers = [
        (
            "numeric_bins",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "bin",
                        KBinsDiscretizer(
                            n_bins=n_bins,
                            encode="onehot",
                            strategy="quantile",
                            subsample=None,
                        ),
                    ),
                ]
            ),
            modeling.NUMERIC_FEATURES,
        ),
        ("categorical", _categorical_one_hot(), modeling.CATEGORICAL_FEATURES),
    ]
    if include_linear:
        transformers.append(
            (
                "numeric_linear",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                modeling.NUMERIC_FEATURES,
            )
        )
    return ColumnTransformer(transformers)


def _ordinal_preprocess() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), modeling.NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
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
                ),
                modeling.CATEGORICAL_FEATURES,
            ),
        ]
    )


def _candidate_models() -> dict[str, Pipeline]:
    candidates: dict[str, Pipeline] = {}
    for c_value in [0.03, 0.1, 0.3, 1.0, 3.0]:
        candidates[f"linear_logistic_c{c_value:g}"] = Pipeline(
            [
                ("preprocess", _linear_preprocess()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        C=c_value,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    for n_bins in [4, 6, 8]:
        for include_linear in [False, True]:
            for c_value in [0.1, 0.5, 1.5]:
                suffix = "hybrid" if include_linear else "bins"
                candidates[f"scorecard_{suffix}_{n_bins}_c{c_value:g}"] = Pipeline(
                    [
                        ("preprocess", _scorecard_preprocess(n_bins, include_linear)),
                        (
                            "model",
                            LogisticRegression(
                                max_iter=3000,
                                class_weight="balanced",
                                C=c_value,
                                random_state=RANDOM_STATE,
                            ),
                        ),
                    ]
                )
    lgbm_settings = [
        (7, 3, 60, 0.02, 500),
        (15, 4, 50, 0.02, 500),
        (23, 5, 45, 0.015, 650),
        (31, 6, 40, 0.012, 750),
    ]
    for leaves, depth, minimum, rate, estimators in lgbm_settings:
        candidates[f"lightgbm_l{leaves}_d{depth}"] = Pipeline(
            [
                ("preprocess", _linear_preprocess()),
                (
                    "model",
                    LGBMClassifier(
                        objective="binary",
                        n_estimators=estimators,
                        learning_rate=rate,
                        num_leaves=leaves,
                        max_depth=depth,
                        min_child_samples=minimum,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.3,
                        reg_lambda=1.5,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    candidates["hist_gradient_conservative"] = Pipeline(
        [
            ("preprocess", _ordinal_preprocess()),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.035,
                    max_iter=350,
                    max_leaf_nodes=15,
                    min_samples_leaf=45,
                    l2_regularization=2.0,
                    class_weight="balanced",
                    categorical_features=list(
                        range(len(modeling.NUMERIC_FEATURES), len(modeling.FEATURES))
                    ),
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    candidates["extra_trees"] = Pipeline(
        [
            ("preprocess", _ordinal_preprocess()),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=600,
                    max_features=0.75,
                    min_samples_leaf=12,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )
    return candidates


def run_search():
    frame = pd.read_csv(PROCESSED_DIR / "account_master.csv")
    train, validation, test = modeling._split_by_customer(frame)
    x_train, y_train = train[modeling.FEATURES], train["target_default"]
    x_validation, y_validation = (
        validation[modeling.FEATURES],
        validation["target_default"],
    )

    fitted = {}
    validation_probability = {}
    result_rows = []
    for name, pipeline in _candidate_models().items():
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_validation)[:, 1]
        threshold = modeling._choose_threshold(y_validation, probability)
        metrics = modeling._metrics(y_validation, probability, threshold)
        fitted[name] = (pipeline, threshold)
        validation_probability[name] = probability
        result_rows.append({"candidate": name, "kind": "single", **metrics})

    # Blend only the top five distinct models. Every blend decision remains a
    # validation decision; no test probabilities are available at this point.
    top_names = [
        row["candidate"]
        for row in sorted(result_rows, key=lambda row: row["ks"], reverse=True)[:5]
    ]
    blend_specs = []
    for left_index, left_name in enumerate(top_names):
        for right_name in top_names[left_index + 1 :]:
            for left_weight in [0.25, 0.5, 0.75]:
                probability = (
                    left_weight * validation_probability[left_name]
                    + (1 - left_weight) * validation_probability[right_name]
                )
                threshold = modeling._choose_threshold(y_validation, probability)
                metrics = modeling._metrics(y_validation, probability, threshold)
                blend_name = f"blend__{left_name}__{right_name}__w{left_weight:g}"
                result_rows.append(
                    {"candidate": blend_name, "kind": "blend", **metrics}
                )
                blend_specs.append(
                    (blend_name, left_name, right_name, left_weight, threshold)
                )

    eligible = [row for row in result_rows if row["recall"] >= 0.70]
    selected_row = max(eligible or result_rows, key=lambda row: (row["ks"], row["pr_auc"]))
    selected_name = selected_row["candidate"]
    x_test, y_test = test[modeling.FEATURES], test["target_default"]

    if selected_row["kind"] == "single":
        selected_model, selected_threshold = fitted[selected_name]
        test_probability = selected_model.predict_proba(x_test)[:, 1]
        selected_spec = {"type": "single", "model": selected_name}
    else:
        spec = next(spec for spec in blend_specs if spec[0] == selected_name)
        _, left_name, right_name, left_weight, selected_threshold = spec
        test_probability = (
            left_weight * fitted[left_name][0].predict_proba(x_test)[:, 1]
            + (1 - left_weight) * fitted[right_name][0].predict_proba(x_test)[:, 1]
        )
        selected_spec = {
            "type": "blend",
            "left_model": left_name,
            "right_model": right_name,
            "left_weight": left_weight,
        }

    test_metrics = modeling._metrics(y_test, test_probability, selected_threshold)
    base_rate = float(y_test.mean())
    cutoff = float(np.quantile(test_probability, 0.80))
    top_prediction = test_probability >= cutoff
    top_precision = float(y_test[top_prediction].mean())
    test_metrics.update(
        {
            "precision_at_top_20_pct": top_precision,
            "recall_at_top_20_pct": float(y_test[top_prediction].sum() / y_test.sum()),
            "precision_lift_at_top_20_pct": top_precision / base_rate,
            "pr_auc_lift_vs_base_rate": test_metrics["pr_auc"] / base_rate,
        }
    )

    results = {
        "guardrails": [
            "Customer-grouped 60/20/20 split retained.",
            "No target-dependent row filtering.",
            "Model and blend selection used validation metrics only.",
            "The test partition was evaluated once after selection.",
        ],
        "selected_candidate": selected_name,
        "selected_specification": selected_spec,
        "validation_metrics": {
            key: selected_row[key]
            for key in ["roc_auc", "pr_auc", "ks", "gini", "precision", "recall", "f1", "threshold"]
        },
        "test_metrics": test_metrics,
        "candidate_count": len(result_rows),
    }
    destination = MODEL_DIR / "experiment_model_search.json"
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(result_rows).sort_values("ks", ascending=False).to_csv(
        MODEL_DIR / "experiment_model_search_validation.csv", index=False
    )
    return results


if __name__ == "__main__":
    print(json.dumps(run_search(), indent=2))
