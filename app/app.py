from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm_assistant import ask_portfolio  # noqa: E402


st.set_page_config(
    page_title="Credit Risk Portfolio Monitor",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2.5rem;}
    [data-testid="stMetric"] {background: white; border: 1px solid #dce5ec; padding: 14px; border-radius: 12px;}
    .hero {background: linear-gradient(120deg,#102a43,#176b68); padding: 24px 28px; border-radius: 16px; color: white; margin-bottom: 18px;}
    .hero h1 {margin: 0 0 6px 0; font-size: 2rem;}
    .hero p {margin: 0; opacity: .9;}
    .note {background:#eef7f5; border-left:4px solid #1f9d8a; padding:12px 14px; border-radius:6px;}
    .risk-high {color:#b42318; font-weight:700;}
    .risk-medium {color:#b54708; font-weight:700;}
    .risk-low {color:#027a48; font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    processed = ROOT / "data" / "processed"
    models = ROOT / "models"
    account = pd.read_csv(processed / "account_scored.csv")
    repayment = pd.read_csv(processed / "repayment_clean.csv")
    anomalies = pd.read_csv(processed / "anomaly_review.csv")
    anomalies["review_decision"] = anomalies["review_decision"].fillna("Pending").astype(str)
    anomalies["review_note"] = anomalies["review_note"].fillna("").astype(str)
    quality = json.loads((processed / "data_quality_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((models / "model_metrics.json").read_text(encoding="utf-8"))
    challenger_path = models / "experiment_model_search.json"
    challenger = (
        json.loads(challenger_path.read_text(encoding="utf-8"))
        if challenger_path.exists()
        else {}
    )
    importance = pd.read_csv(models / "feature_importance.csv")
    return account, repayment, anomalies, quality, metrics, challenger, importance


account, repayment, anomalies, quality, model_metrics, challenger_metrics, importance = load_data()

st.sidebar.title("Risk Monitor")
page = st.sidebar.radio(
    "Navigate",
    ["Portfolio", "Account review", "Ask the Portfolio", "Anomaly review", "Methodology"],
)
st.sidebar.caption("Synthetic portfolio · 14,685 accounts")


def header(title: str, subtitle: str):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def filtered_portfolio(frame: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("### Portfolio filters")
    selections = {
        "loan_type": st.sidebar.multiselect("Loan type", sorted(frame.loan_type.unique())),
        "region": st.sidebar.multiselect("Region", sorted(frame.region.unique())),
        "employment_type": st.sidebar.multiselect(
            "Employment", sorted(frame.employment_type.unique())
        ),
        "risk_band": st.sidebar.multiselect("Risk band", ["High", "Medium", "Low"]),
    }
    result = frame
    for column, values in selections.items():
        if values:
            result = result[result[column].isin(values)]
    return result


if page == "Portfolio":
    header("Credit Risk Portfolio Monitor", "Exposure, default risk and early-warning signals in one view")
    data = filtered_portfolio(account)
    if data.empty:
        st.warning("No accounts match the selected filters.")
        st.stop()

    cols = st.columns(5)
    cols[0].metric("Accounts", f"{len(data):,}")
    cols[1].metric("Customers", f"{data.customer_id.nunique():,}")
    cols[2].metric("Original exposure", f"${data.loan_amount.sum()/1e6:,.1f}M")
    cols[3].metric("Default rate", f"{data.target_default.mean():.1%}")
    cols[4].metric("High-risk accounts", f"{data.risk_band.eq('High').mean():.1%}")

    left, right = st.columns([1.1, 1])
    with left:
        risk = data.groupby("risk_band", as_index=False).agg(
            accounts=("account_id", "count"), exposure=("latest_outstanding_balance", "sum")
        )
        risk["risk_band"] = pd.Categorical(risk["risk_band"], ["Low", "Medium", "High"], ordered=True)
        risk = risk.sort_values("risk_band")
        fig = px.bar(
            risk,
            x="risk_band",
            y="accounts",
            color="risk_band",
            color_discrete_map={"Low": "#2a9d8f", "Medium": "#f4a261", "High": "#d1495b"},
            title="Accounts by model risk band",
            text_auto=",.0f",
        )
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Accounts")
        st.plotly_chart(fig, width="stretch")
    with right:
        by_loan = data.groupby("loan_type", as_index=False).agg(
            default_rate=("target_default", "mean"), accounts=("account_id", "count")
        ).sort_values("default_rate")
        fig = px.bar(
            by_loan,
            x="default_rate",
            y="loan_type",
            orientation="h",
            color="default_rate",
            color_continuous_scale=["#b8e0d2", "#d1495b"],
            title="Default rate by loan type",
            text=by_loan["default_rate"].map(lambda value: f"{value:.1%}"),
        )
        fig.update_layout(coloraxis_showscale=False, xaxis_tickformat=".0%", yaxis_title=None)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Cross-dimensional risk cut")
    heat = data.pivot_table(
        index="employment_type", columns="loan_type", values="target_default", aggfunc="mean"
    )
    fig = px.imshow(
        heat,
        text_auto=".1%",
        color_continuous_scale=["#edf8f5", "#f4a261", "#b42318"],
        aspect="auto",
        title="Default rate: employment type × loan type",
    )
    fig.update_layout(coloraxis_colorbar_tickformat=".0%", xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        income = data.groupby("income_band", as_index=False).agg(
            default_rate=("target_default", "mean"), accounts=("account_id", "count")
        )
        income["income_band"] = pd.Categorical(
            income["income_band"], ["Q1 Lowest", "Q2", "Q3", "Q4", "Q5 Highest"], ordered=True
        )
        income = income.sort_values("income_band")
        fig = px.line(income, x="income_band", y="default_rate", markers=True, title="Default rate by income quintile")
        fig.update_yaxes(tickformat=".0%")
        fig.update_xaxes(title=None)
        st.plotly_chart(fig, width="stretch")
    with right:
        trend = data.groupby("origination_year", as_index=False).agg(
            default_rate=("target_default", "mean"), exposure=("loan_amount", "sum")
        )
        fig = px.line(trend, x="origination_year", y="default_rate", markers=True, title="Origination-cohort default trend")
        fig.update_yaxes(tickformat=".0%")
        fig.update_xaxes(dtick=1, title=None)
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        age_order = ["Under 25", "25–34", "35–44", "45–54", "55–64", "65+"]
        age = data.groupby("age_band", as_index=False).agg(
            default_rate=("target_default", "mean"), accounts=("account_id", "count")
        )
        age["age_band"] = pd.Categorical(age["age_band"], age_order, ordered=True)
        age = age.sort_values("age_band")
        fig = px.bar(
            age,
            x="age_band",
            y="default_rate",
            text=age["default_rate"].map(lambda value: f"{value:.1%}"),
            title="Default rate by age band",
            color="default_rate",
            color_continuous_scale=["#b8e0d2", "#d1495b"],
        )
        fig.update_layout(coloraxis_showscale=False, xaxis_title=None)
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
    with right:
        region = data.groupby("region", as_index=False).agg(
            default_rate=("target_default", "mean"), accounts=("account_id", "count")
        ).sort_values("default_rate")
        fig = px.bar(
            region,
            x="default_rate",
            y="region",
            orientation="h",
            text=region["default_rate"].map(lambda value: f"{value:.1%}"),
            title="Default rate by region",
            color="default_rate",
            color_continuous_scale=["#b8e0d2", "#d1495b"],
        )
        fig.update_layout(coloraxis_showscale=False, xaxis_tickformat=".0%", yaxis_title=None)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Portfolio insights and actions")
    loan_signal = data.groupby("loan_type").target_default.agg(["mean", "size"]).sort_values("mean", ascending=False)
    employment_signal = data.groupby("employment_type").target_default.agg(["mean", "size"]).sort_values("mean", ascending=False)
    region_signal = data.groupby("region").target_default.agg(["mean", "size"]).sort_values("mean", ascending=False)
    age_signal = data.groupby("age_band").target_default.agg(["mean", "size"]).sort_values("mean", ascending=False)
    income_signal = data.groupby("income_band").target_default.mean()
    cross_all = (
        data.groupby(["employment_type", "loan_type"])
        .target_default.agg(["mean", "size"])
        .sort_values("mean", ascending=False)
    )
    cross_signal = cross_all.loc[cross_all["size"] >= 50]
    if cross_signal.empty:
        cross_signal = cross_all
    cross_employment, cross_loan = cross_signal.index[0]
    insights = [
        (
            f"{loan_signal.index[0]} has the highest loan-type default rate at {loan_signal.iloc[0]['mean']:.1%} "
            f"across {int(loan_signal.iloc[0]['size']):,} accounts.",
            "Review product underwriting, pricing and early-warning limits for this product first.",
        ),
        (
            f"The lowest-income quintile defaults at {income_signal.get('Q1 Lowest', float('nan')):.1%}, compared with "
            f"{income_signal.get('Q5 Highest', float('nan')):.1%} in the highest-income quintile.",
            "Strengthen affordability monitoring and prioritize proactive assistance for lower-income borrowers.",
        ),
        (
            f"{employment_signal.index[0]} borrowers have the highest employment-segment default rate at "
            f"{employment_signal.iloc[0]['mean']:.1%}.",
            "Use this segment for targeted monitoring, while retaining human review and fair-lending controls.",
        ),
        (
            f"{age_signal.index[0]} is the highest-risk age band at {age_signal.iloc[0]['mean']:.1%}; "
            f"{region_signal.index[0]} is the highest-risk region at {region_signal.iloc[0]['mean']:.1%}.",
            "Track both cuts for drift and capacity planning rather than applying age or geography as an automatic adverse-action rule.",
        ),
        (
            f"The highest-risk employment × loan combination with at least 50 accounts is "
            f"{cross_employment} / {cross_loan} at {cross_signal.iloc[0]['mean']:.1%} "
            f"({int(cross_signal.iloc[0]['size']):,} accounts).",
            "Create a focused review queue for this intersection and validate whether the pattern persists out of sample.",
        ),
    ]
    for number, (finding, action) in enumerate(insights, 1):
        st.markdown(f"**{number}. Finding:** {finding}  \n**So what:** {action}")

elif page == "Account review":
    header("Account review", "Model score, drivers, repayment trajectory and bureau context")
    default_account = str(account.sort_values("risk_score", ascending=False).iloc[0].account_id)
    selected = st.selectbox(
        "Select account",
        options=account.account_id.astype(str).sort_values().tolist(),
        index=account.account_id.astype(str).sort_values().tolist().index(default_account),
    )
    row = account.loc[account.account_id.astype(str).eq(selected)].iloc[0]
    repayment_account = repayment.loc[repayment.account_id.astype(str).eq(selected)].copy()

    cols = st.columns(5)
    cols[0].metric("Risk score", f"{row.risk_score:.1f}/100")
    cols[1].metric("Risk band", row.risk_band)
    cols[2].metric("Loan type", row.loan_type)
    cols[3].metric("Outstanding", f"${row.latest_outstanding_balance:,.0f}")
    cols[4].metric("Observed outcome", "Default" if row.target_default else "Performing")

    st.markdown(f"**Key risk indicators:** {row.key_risk_indicators}")
    left, right = st.columns([1.5, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=repayment_account.payment_month, y=repayment_account.days_past_due, name="Days past due", marker_color="#d1495b"))
        fig.add_trace(go.Scatter(x=repayment_account.payment_month, y=repayment_account.payment_ratio * 100, name="Payment ratio %", yaxis="y2", line=dict(color="#1f9d8a", width=3)))
        fig.update_layout(
            title="Twelve-month repayment behaviour",
            xaxis_title="Payment month",
            yaxis_title="Days past due",
            yaxis2=dict(title="Payment ratio %", overlaying="y", side="right", range=[0, 110]),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("#### Customer and bureau summary")
        summary = pd.DataFrame(
            {
                "Measure": [
                    "Annual income", "Employment", "Region", "External lines", "Written-off lines",
                    "Bureau utilization", "Maximum bureau DPD", "First-10-month missed payments",
                ],
                "Value": [
                    f"${row.annual_income:,.0f}", row.employment_type, row.region,
                    f"{row.bureau_line_count:,.0f}", f"{row.bureau_written_off_count:,.0f}",
                    f"{row.bureau_total_utilization:.1%}", f"{row.bureau_max_dpd:,.0f}",
                    f"{row.missed_months_10m:,.0f}",
                ],
            }
        )
        st.dataframe(summary, hide_index=True, width="stretch")

elif page == "Ask the Portfolio":
    header("Ask the Portfolio", "Grounded natural-language analysis with exact, backend-calculated figures")
    st.markdown(
        '<div class="note">The LLM only translates your question into an approved metric, grouping and filters. '
        'All numbers are calculated by application code from the scored account table.</div>',
        unsafe_allow_html=True,
    )
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
        secret_model = st.secrets.get("OPENAI_MODEL", "")
    except Exception:
        secret_key = ""
        secret_model = ""
    api_key = secret_key or os.getenv("OPENAI_API_KEY", "")
    model = secret_model or os.getenv("OPENAI_MODEL", "gpt-6-luna")
    examples = [
        "Which loan type has the highest default rate?",
        "Show the default rate by employment type for South region.",
        "What is the outstanding exposure for high-risk accounts?",
    ]
    st.caption("Try: " + " · ".join(examples))
    question = st.text_input("Portfolio question", placeholder=examples[0])
    if st.button("Ask", type="primary"):
        if not api_key:
            st.warning("The assistant is disabled because OPENAI_API_KEY is not configured. The rest of the dashboard remains available.")
        elif not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Translating and validating the portfolio query…"):
                    result = ask_portfolio(question, account, api_key=api_key, model=model)
                st.success(result.answer)
                if not result.table.empty:
                    display = result.table.copy()
                    if "value" in display:
                        display["value"] = display["value"].round(4)
                    st.dataframe(display, hide_index=True, width="stretch")
                st.caption(result.provenance)
            except Exception as exc:
                st.error(f"The question could not be processed safely: {exc}")

    with st.expander("Controls and human review"):
        st.markdown(
            "**Trigger:** an analyst submits a dashboard question.  \n"
            "**False positive:** a query is accepted but mapped to the wrong approved metric or filter.  \n"
            "**False negative:** an answerable portfolio question is declined.  \n"
            "**Human step:** verify the displayed provenance and figures before using the answer in a decision."
        )

elif page == "Anomaly review":
    header("Anomaly review", "Peer-relative and historical pattern alerts with analyst disposition")
    cols = st.columns(4)
    cols[0].metric("Flagged accounts", f"{len(anomalies):,}")
    cols[1].metric("Flag rate", f"{len(anomalies)/len(account):.1%}")
    cols[2].metric("High-risk among flags", f"{anomalies.risk_band.eq('High').mean():.1%}")
    cols[3].metric("Median anomaly score", f"{anomalies.anomaly_score.median():.1f}")

    if "anomaly_editor" not in st.session_state:
        st.session_state.anomaly_editor = anomalies.copy()
    risk_filter = st.multiselect("Risk-band filter", ["High", "Medium", "Low"], default=["High", "Medium"])
    editor_data = st.session_state.anomaly_editor
    if risk_filter:
        editor_data = editor_data[editor_data.risk_band.isin(risk_filter)]
    edited = st.data_editor(
        editor_data.head(250),
        hide_index=True,
        width="stretch",
        disabled=[column for column in editor_data.columns if column not in ["review_decision", "review_note"]],
        column_config={
            "review_decision": st.column_config.SelectboxColumn(
                "Decision", options=["Pending", "Confirm", "Dismiss", "Escalate"], required=True
            ),
            "review_note": st.column_config.TextColumn("Analyst note"),
            "anomaly_score": st.column_config.ProgressColumn("Anomaly score", min_value=0, max_value=100),
        },
        key="anomaly_table",
    )
    if st.button("Save decisions for this session"):
        updated = st.session_state.anomaly_editor.set_index("account_id")
        edits = edited.set_index("account_id")
        updated.update(edits[["review_decision", "review_note"]])
        st.session_state.anomaly_editor = updated.reset_index()
        st.success("Review decisions saved in this dashboard session.")

    with st.expander("Trigger, errors and human review"):
        st.markdown(
            "**Trigger:** top multivariate anomaly scores or transparent rules such as repeated problematic months and multiple bureau write-offs.  \n"
            "**False positive:** legitimate but unusual repayment or bureau behaviour.  \n"
            "**False negative:** emerging risk that resembles normal peer behaviour.  \n"
            "**Human step:** confirm, dismiss or escalate after reviewing the account history; production decisions require durable audit storage."
        )

else:
    header("Methodology and controls", "Model performance, data quality and production monitoring")
    test = model_metrics["test"]
    cols = st.columns(6)
    for column, key, fmt in [
        (cols[0], "ROC AUC", "roc_auc"),
        (cols[1], "KS", "ks"),
        (cols[2], "Gini", "gini"),
        (cols[3], "Precision", "precision"),
        (cols[4], "Recall", "recall"),
        (cols[5], "F1", "f1"),
    ]:
        column.metric(key, f"{test[fmt]:.3f}")
    st.caption(
        f"Selected model: {model_metrics['selected_model']} · threshold {test['threshold']:.2f} · "
        f"customer-grouped test set of {model_metrics['test_accounts']:,} accounts"
    )

    if challenger_metrics:
        challenger_test = challenger_metrics["test_metrics"]
        comparison = pd.DataFrame(
            [
                {
                    "Model": "Production logistic regression",
                    "KS": test["ks"],
                    "PR-AUC": test["pr_auc"],
                    "Precision": test["precision"],
                    "Recall": test["recall"],
                    "F1": test["f1"],
                },
                {
                    "Model": "Extra Trees / logistic blend challenger",
                    "KS": challenger_test["ks"],
                    "PR-AUC": challenger_test["pr_auc"],
                    "Precision": challenger_test["precision"],
                    "Recall": challenger_test["recall"],
                    "F1": challenger_test["f1"],
                },
            ]
        )
        st.markdown("#### Why logistic regression is production")
        st.dataframe(
            comparison.style.format(
                {"KS": "{:.3f}", "PR-AUC": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "The blend improves KS from 0.285 to 0.296, but logistic regression retains higher PR-AUC, recall and F1. "
            "For portfolio monitoring, that is the better operating trade-off and is easier to explain and govern."
        )

    left, right = st.columns([1.15, 1])
    with left:
        top = importance.head(12).sort_values("importance_mean")
        fig = px.bar(top, x="importance_mean", y="feature", orientation="h", title="Permutation importance on validation data")
        fig.update_layout(xaxis_title="Change in average precision", yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("#### Data-quality decisions")
        quality_table = pd.DataFrame(
            {
                "Issue": [
                    "Gender labels standardized", "Implausible employment tenure", "Customer-since chronology",
                    "Pre-origination collection events", "Future bureau close dates", "Customers without bureau lines",
                ],
                "Rows": [
                    quality["gender_labels_standardized"], quality["employment_tenure_implausible"],
                    quality["customer_since_after_origination"], quality["collections_before_origination"],
                    quality["bureau_future_close_dates"], quality["bureau_customers_missing"],
                ],
                "Treatment": [
                    "Mapped M/F to Male/Female", "Flagged and median-imputed", "Flagged; excluded from tenure features",
                    "Flagged; excluded from time analysis/model", "Flagged; excluded from recency logic", "Zero-count bureau features",
                ],
            }
        )
        st.dataframe(quality_table, hide_index=True, width="stretch")

    st.markdown("#### Model scope and monitoring")
    st.markdown(
        "The model is an early-warning portfolio-monitoring classifier. It uses repayment months 1–10; months 11–12 and collections are excluded from model features. "
        "Production monitoring should track AUC, KS, Gini, precision/recall/F1 at the operating threshold, calibration, feature and score PSI, missingness, segment performance, analyst overrides and API availability."
    )
