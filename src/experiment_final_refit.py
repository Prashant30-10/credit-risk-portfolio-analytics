"""Refit the validation-selected blend on the full development population."""

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

import src.modeling as modeling
from src.config import MODEL_DIR, PROCESSED_DIR
from src.experiment_model_search import _candidate_models


def run_refit():
    search = json.loads(
        (MODEL_DIR / "experiment_model_search.json").read_text(encoding="utf-8")
    )
    specification = search["selected_specification"]
    if specification["type"] != "blend":
        raise ValueError("The selected search candidate is not a blend.")

    frame = pd.read_csv(PROCESSED_DIR / "account_master.csv")
    train, validation, test = modeling._split_by_customer(frame)
    development = pd.concat([train, validation], ignore_index=True)

    candidates = _candidate_models()
    left_name = specification["left_model"]
    right_name = specification["right_model"]
    left_weight = float(specification["left_weight"])
    left_model = candidates[left_name]
    right_model = candidates[right_name]
    left_model.fit(development[modeling.FEATURES], development["target_default"])
    right_model.fit(development[modeling.FEATURES], development["target_default"])

    probability = (
        left_weight * left_model.predict_proba(test[modeling.FEATURES])[:, 1]
        + (1 - left_weight) * right_model.predict_proba(test[modeling.FEATURES])[:, 1]
    )
    threshold = float(search["validation_metrics"]["threshold"])
    metrics = modeling._metrics(test["target_default"], probability, threshold)
    base_rate = float(test["target_default"].mean())
    top_cutoff = float(np.quantile(probability, 0.80))
    top_mask = probability >= top_cutoff
    top_precision = float(test.loc[top_mask, "target_default"].mean())
    metrics.update(
        {
            "precision_at_top_20_pct": top_precision,
            "recall_at_top_20_pct": float(
                test.loc[top_mask, "target_default"].sum()
                / test["target_default"].sum()
            ),
            "precision_lift_at_top_20_pct": top_precision / base_rate,
            "pr_auc_lift_vs_base_rate": metrics["pr_auc"] / base_rate,
        }
    )
    output = {
        "status": "Final refit of the frozen validation-selected specification.",
        "development_accounts": len(development),
        "test_accounts": len(test),
        "selected_specification": specification,
        "threshold_frozen_from_validation": threshold,
        "test_metrics": metrics,
    }
    (MODEL_DIR / "experiment_final_refit.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    joblib.dump(
        {
            "left_model": left_model,
            "right_model": right_model,
            "left_weight": left_weight,
            "features": modeling.FEATURES,
            "threshold": threshold,
            "left_name": left_name,
            "right_name": right_name,
        },
        MODEL_DIR / "experiment_final_refit.joblib",
    )
    return output


if __name__ == "__main__":
    print(json.dumps(run_refit(), indent=2))
