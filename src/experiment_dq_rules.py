"""Compare auditable, target-independent data-quality exclusion rules."""

from __future__ import annotations

import json
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import src.modeling as modeling
from src.config import MODEL_DIR, PROCESSED_DIR


def _severity(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["interest_rate_pct_missing"].fillna(0)
        + frame["years_employed_missing"].fillna(0)
        + frame["num_dependents_missing"].fillna(0)
        + frame["years_at_residence_missing"].fillna(0)
        + frame["employment_tenure_implausible"].fillna(0)
        + frame["bureau_dpd_missing_count"].fillna(0).gt(0).astype(int)
        + frame["bureau_future_close_count"].fillna(0).ge(2).astype(int)
        + frame["invalid_collection_events"].fillna(0).ge(2).astype(int)
    )


def _keep_by_rule(frame: pd.DataFrame, rule: str) -> pd.Series:
    if rule == "none":
        return pd.Series(True, index=frame.index)
    direct_rules = {
        "interest_rate_missing": frame["interest_rate_pct_missing"].eq(0),
        "years_employed_missing": frame["years_employed_missing"].eq(0),
        "dependents_missing": frame["num_dependents_missing"].eq(0),
        "residence_missing": frame["years_at_residence_missing"].eq(0),
        "employment_tenure_implausible": frame["employment_tenure_implausible"].eq(0),
        "bureau_dpd_missing": frame["bureau_dpd_missing_count"].fillna(0).eq(0),
        "multiple_future_bureau_dates": frame["bureau_future_close_count"].fillna(0).lt(2),
        "multiple_invalid_collections": frame["invalid_collection_events"].fillna(0).lt(2),
    }
    if rule in direct_rules:
        return direct_rules[rule]
    if rule.startswith("severity_top_"):
        fraction = float(rule.rsplit("_", 1)[-1]) / 100
        number_to_drop = int(np.floor(len(frame) * fraction))
        severity = _severity(frame)
        # Stable account identifier is used only to resolve equal-severity ties.
        tie_breaker = pd.util.hash_pandas_object(
            frame["account_id"].astype(str), index=False
        ).astype("uint64")
        order = pd.DataFrame(
            {"severity": severity, "tie": tie_breaker}, index=frame.index
        ).sort_values(["severity", "tie"], ascending=[False, True])
        keep = pd.Series(True, index=frame.index)
        keep.loc[order.index[:number_to_drop]] = False
        return keep
    raise KeyError(rule)


def run_experiment():
    frame = pd.read_csv(PROCESSED_DIR / "account_master.csv")
    train, validation, test = modeling._split_by_customer(frame)
    rules = [
        "none",
        "interest_rate_missing",
        "years_employed_missing",
        "dependents_missing",
        "residence_missing",
        "employment_tenure_implausible",
        "bureau_dpd_missing",
        "multiple_future_bureau_dates",
        "multiple_invalid_collections",
        "severity_top_5",
        "severity_top_7.5",
        "severity_top_10",
    ]
    validation_rows = []
    fitted = {}
    for rule in rules:
        train_filtered = train.loc[_keep_by_rule(train, rule)].copy()
        validation_filtered = validation.loc[_keep_by_rule(validation, rule)].copy()
        test_filtered = test.loc[_keep_by_rule(test, rule)].copy()
        train_exclusion = 1 - len(train_filtered) / len(train)
        validation_exclusion = 1 - len(validation_filtered) / len(validation)
        test_exclusion = 1 - len(test_filtered) / len(test)
        if max(train_exclusion, validation_exclusion, test_exclusion) > 0.1001:
            continue
        model = modeling._logistic_pipeline()
        model.fit(train_filtered[modeling.FEATURES], train_filtered["target_default"])
        probability = model.predict_proba(validation_filtered[modeling.FEATURES])[:, 1]
        threshold = modeling._choose_threshold(
            validation_filtered["target_default"], probability
        )
        metrics = modeling._metrics(
            validation_filtered["target_default"], probability, threshold
        )
        validation_rows.append(
            {
                "rule": rule,
                "train_exclusion_rate": train_exclusion,
                "validation_exclusion_rate": validation_exclusion,
                "test_exclusion_rate": test_exclusion,
                **metrics,
            }
        )
        fitted[rule] = (model, threshold, test_filtered)

    eligible = [row for row in validation_rows if row["recall"] >= 0.70]
    selected = max(eligible or validation_rows, key=lambda row: (row["ks"], row["pr_auc"]))
    model, threshold, test_filtered = fitted[selected["rule"]]
    test_probability = model.predict_proba(test_filtered[modeling.FEATURES])[:, 1]
    test_metrics = modeling._metrics(
        test_filtered["target_default"], test_probability, threshold
    )
    output = {
        "guardrails": [
            "Rules use data-quality fields only, never the target.",
            "Every retained candidate excludes no more than 10% from any split.",
            "Rule selection uses validation KS only, with a 70% recall floor.",
            "The selected rule is evaluated once on the test partition.",
        ],
        "selected_rule": selected["rule"],
        "selected_validation_metrics": selected,
        "test_rows": len(test_filtered),
        "test_metrics": test_metrics,
        "candidate_count": len(validation_rows),
    }
    (MODEL_DIR / "experiment_dq_rules.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    pd.DataFrame(validation_rows).sort_values("ks", ascending=False).to_csv(
        MODEL_DIR / "experiment_dq_rules_validation.csv", index=False
    )
    return output


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))
