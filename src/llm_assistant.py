from __future__ import annotations

import json
from typing import Literal

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

from src.analytics import DIMENSIONS, METRICS, QueryResult, category_catalog, execute_query


MetricName = Literal[
    "default_rate",
    "account_count",
    "customer_count",
    "original_exposure",
    "outstanding_exposure",
    "average_risk_score",
    "delinquency_rate",
]

DimensionName = Literal[
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


class FilterSpec(BaseModel):
    field: DimensionName
    value: str


class PortfolioQuerySpec(BaseModel):
    answerable: bool = Field(description="Whether the question can be answered using the approved schema.")
    decline_reason: str | None = Field(description="Reason when the question is not answerable.")
    metric: MetricName | None = Field(description="Approved metric to calculate.")
    group_by: DimensionName | None = Field(description="Optional approved grouping dimension.")
    filters: list[FilterSpec] = Field(description="Zero or more exact categorical filters.")
    top_n: int = Field(ge=1, le=20, description="Number of grouped rows to return.")


def ask_portfolio(
    question: str,
    frame: pd.DataFrame,
    api_key: str,
    model: str = "gpt-6-luna",
) -> QueryResult:
    catalog = category_catalog(frame)
    instructions = f"""
You translate portfolio questions into a restricted analytical query.
Never calculate or invent a number. The application calculates all figures.
Approved metrics: {json.dumps(METRICS)}
Approved dimensions: {json.dumps(DIMENSIONS)}
Available categorical values: {json.dumps(catalog)}
Return answerable=false when the question needs unsupported data, free text,
external knowledge, personal advice, causality, or a metric outside this schema.
Use exact available category values in filters. For 'highest' or 'lowest' questions,
set group_by to the relevant dimension. Use top_n=5 unless the user requests another count.
""".strip()
    client = OpenAI(api_key=api_key)
    response = client.responses.parse(
        model=model,
        instructions=instructions,
        input=question,
        text_format=PortfolioQuerySpec,
        store=False,
    )
    spec = response.output_parsed
    if spec is None:
        raise ValueError("The assistant did not return a usable query plan.")
    if not spec.answerable or spec.metric is None:
        reason = spec.decline_reason or "The requested information is not available in this portfolio dataset."
        return QueryResult(
            answer=f"I can’t answer that from the available dashboard data. {reason}",
            table=pd.DataFrame(),
            provenance="No portfolio query was executed.",
        )
    return execute_query(
        frame,
        metric=spec.metric,
        group_by=spec.group_by,
        filters=[item.model_dump() for item in spec.filters],
        top_n=spec.top_n,
    )
