from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.config import PROCESSED_DIR, RANDOM_STATE


ANOMALY_FEATURES = [
    "late_months_12m",
    "missed_months_12m",
    "max_dpd_12m",
    "mean_dpd_12m",
    "dpd_volatility_12m",
    "max_problem_streak_12m",
    "mean_payment_ratio_12m",
    "bureau_total_utilization",
    "bureau_max_dpd",
    "bureau_written_off_count",
]


def _robust_peer_z(frame: pd.DataFrame, feature: str) -> pd.Series:
    median = frame.groupby("loan_type")[feature].transform("median")
    absolute = (frame[feature] - median).abs()
    mad = absolute.groupby(frame["loan_type"]).transform("median").replace(0, np.nan)
    return ((frame[feature] - median) / (1.4826 * mad)).replace([np.inf, -np.inf], np.nan).fillna(0)


def detect_anomalies(
    scored_path=PROCESSED_DIR / "account_scored.csv",
    output_path=PROCESSED_DIR / "anomaly_review.csv",
) -> pd.DataFrame:
    frame = pd.read_csv(scored_path)
    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            (
                "model",
                IsolationForest(
                    n_estimators=300,
                    contamination=0.08,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipeline.fit(frame[ANOMALY_FEATURES])
    raw_score = -pipeline.decision_function(frame[ANOMALY_FEATURES])
    frame["anomaly_score"] = pd.Series(raw_score).rank(pct=True).mul(100).round(1)
    frame["peer_z_max_dpd"] = _robust_peer_z(frame, "max_dpd_12m")
    frame["peer_z_missed"] = _robust_peer_z(frame, "missed_months_12m")
    frame["peer_z_utilization"] = _robust_peer_z(frame, "bureau_total_utilization")

    rule_flag = (
        frame["peer_z_max_dpd"].gt(3)
        | frame["peer_z_missed"].gt(3)
        | frame["peer_z_utilization"].gt(3)
        | frame["max_problem_streak_12m"].ge(3)
        | frame["bureau_written_off_count"].ge(2)
    )
    frame["is_anomaly"] = (frame["anomaly_score"].ge(92) | rule_flag)

    def reasons(row: pd.Series) -> str:
        output = []
        if row["peer_z_max_dpd"] > 3:
            output.append("DPD is unusually high for this loan type")
        if row["peer_z_missed"] > 3:
            output.append("missed-payment count is unusually high versus peers")
        if row["peer_z_utilization"] > 3:
            output.append("bureau utilization is unusually high versus peers")
        if row["max_problem_streak_12m"] >= 3:
            output.append("three-or-more consecutive problematic months")
        if row["bureau_written_off_count"] >= 2:
            output.append("multiple external written-off lines")
        if not output and row["anomaly_score"] >= 92:
            output.append("unusual multivariate repayment/bureau pattern")
        return "; ".join(output)

    frame["anomaly_reason"] = frame.apply(reasons, axis=1)
    frame["review_decision"] = "Pending"
    frame["review_note"] = ""
    review = frame.loc[
        frame["is_anomaly"],
        [
            "account_id",
            "customer_id",
            "loan_type",
            "risk_score",
            "risk_band",
            "anomaly_score",
            "anomaly_reason",
            "max_dpd_12m",
            "missed_months_12m",
            "bureau_total_utilization",
            "review_decision",
            "review_note",
        ],
    ].sort_values("anomaly_score", ascending=False)
    review.to_csv(output_path, index=False)
    return review


if __name__ == "__main__":
    result = detect_anomalies()
    print(f"Flagged {len(result):,} accounts for review")
