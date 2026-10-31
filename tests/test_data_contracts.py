import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_account_grain_and_target():
    account = pd.read_csv(ROOT / "data" / "processed" / "account_scored.csv")
    assert len(account) == 14685
    assert account["account_id"].is_unique
    assert account["target_default"].isin([0, 1]).all()
    assert 0.15 < account["target_default"].mean() < 0.17


def test_model_outputs_are_complete():
    metrics = json.loads((ROOT / "models" / "model_metrics.json").read_text(encoding="utf-8"))
    for name in ["roc_auc", "pr_auc", "ks", "gini", "precision", "recall", "f1"]:
        assert name in metrics["test"]
        assert 0 <= metrics["test"][name] <= 1
    assert metrics["test"]["recall"] >= 0.70


def test_anomaly_queue_has_explanations():
    anomalies = pd.read_csv(ROOT / "data" / "processed" / "anomaly_review.csv")
    assert not anomalies.empty
    assert anomalies["anomaly_reason"].fillna("").str.len().gt(0).all()

