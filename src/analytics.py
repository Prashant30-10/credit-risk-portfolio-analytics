from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


METRICS = {
    "default_rate": "Default rate",
    "account_count": "Account count",
    "customer_count": "Customer count",
    "original_exposure": "Original exposure",
    "outstanding_exposure": "Outstanding exposure",
    "average_risk_score": "Average model risk score",
    "delinquency_rate": "12-month delinquency rate",
}

DIMENSIONS = [
    "loan_type",
    "employment_type",
    "region",
    "age_band",
    "income_band",
    "risk_band",
    "disbursal_channel",
    "gender",
    "city_tier",
    "origination_year",
]


@dataclass
class QueryResult:
    answer: str
    table: pd.DataFrame
    provenance: str


def _metric_series(grouped, metric: str):
    if metric == "default_rate":
        return grouped["target_default"].mean()
    if metric == "account_count":
        return grouped["account_id"].count()
    if metric == "customer_count":
        return grouped["customer_id"].nunique()
    if metric == "original_exposure":
        return grouped["loan_amount"].sum()
    if metric == "outstanding_exposure":
        return grouped["latest_outstanding_balance"].sum()
    if metric == "average_risk_score":
        return grouped["risk_score"].mean()
    if metric == "delinquency_rate":
        return grouped["is_delinquent_12m"].mean()
    raise ValueError(f"Unsupported metric: {metric}")


def _format_value(metric: str, value: float) -> str:
    if metric in {"default_rate", "delinquency_rate"}:
        return f"{value:.1%}"
    if metric in {"original_exposure", "outstanding_exposure"}:
        return f"${value:,.0f}"
    if metric == "average_risk_score":
        return f"{value:.1f}/100"
    return f"{value:,.0f}"


def execute_query(
    frame: pd.DataFrame,
    metric: str,
    group_by: str | None,
    filters: list[dict[str, str]] | None = None,
    top_n: int = 5,
) -> QueryResult:
    if metric not in METRICS:
        raise ValueError("The requested metric is not approved.")
    if group_by is not None and group_by not in DIMENSIONS:
        raise ValueError("The requested grouping is not approved.")

    data = frame.copy()
    data["is_delinquent_12m"] = data["max_dpd_12m"].gt(0)
    applied = []
    for item in filters or []:
        field, requested = item["field"], str(item["value"])
        if field not in DIMENSIONS:
            raise ValueError(f"Filtering by {field} is not approved.")
        values = {str(value).casefold(): value for value in data[field].dropna().unique()}
        if requested.casefold() not in values:
            raise ValueError(f"No value '{requested}' exists for {field}.")
        actual = values[requested.casefold()]
        data = data[data[field].eq(actual)]
        applied.append(f"{field}={actual}")
    if data.empty:
        raise ValueError("No records match the requested filters.")

    if group_by:
        values = _metric_series(data.groupby(group_by, dropna=False), metric)
        result = values.rename("value").reset_index().sort_values("value", ascending=False)
        result = result.head(max(1, min(int(top_n), 20)))
        leader = result.iloc[0]
        answer = (
            f"{leader[group_by]} has the highest {METRICS[metric].lower()} at "
            f"{_format_value(metric, float(leader['value']))}. "
            f"This is based on {len(data):,} matching account records."
        )
    else:
        grouped = data.assign(_all="Portfolio").groupby("_all")
        value = float(_metric_series(grouped, metric).iloc[0])
        result = pd.DataFrame({"metric": [METRICS[metric]], "value": [value]})
        answer = (
            f"The {METRICS[metric].lower()} is {_format_value(metric, value)}, "
            f"based on {len(data):,} matching account records."
        )
    provenance = "Source: data/processed/account_scored.csv"
    if applied:
        provenance += "; filters: " + ", ".join(applied)
    provenance += f"; metric: {metric}"
    if group_by:
        provenance += f"; grouped by: {group_by}"
    return QueryResult(answer=answer, table=result, provenance=provenance)


def category_catalog(frame: pd.DataFrame) -> dict[str, list[str]]:
    return {
        dimension: sorted(str(value) for value in frame[dimension].dropna().unique())
        for dimension in DIMENSIONS
    }

