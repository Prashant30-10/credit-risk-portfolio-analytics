from __future__ import annotations

import json
import textwrap

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.config import DOCS_DIR, MODEL_DIR, PROCESSED_DIR


NAVY = "#102A43"
TEAL = "#1F9D8A"
PALE = "#EAF4F2"
ORANGE = "#F4A261"
RED = "#C94C5C"
GREY = "#526777"


def _wrap_text(text: str, width: int) -> str:
    return "\n".join(textwrap.fill(line, width) if line else "" for line in text.splitlines())


def _box(
    ax,
    xy,
    width,
    height,
    title,
    body,
    color=TEAL,
    *,
    title_size=10.5,
    body_size=7.8,
    body_wrap=29,
):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.4, edgecolor=color, facecolor="white",
    )
    ax.add_patch(patch)
    title_lines = title.count("\n") + 1
    ax.text(x + 0.02, y + height - 0.05, title, fontsize=title_size, fontweight="bold", color=NAVY, va="top", linespacing=1.1)
    body_top = y + height - 0.12 - 0.032 * (title_lines - 1)
    ax.text(x + 0.02, body_top, _wrap_text(body, body_wrap), fontsize=body_size, color=GREY, va="top", linespacing=1.3)


def _arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.4, color=TEAL))


def architecture_pdf():
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#F7F9FB")
    ax.text(0.05, 0.94, "Credit Risk Portfolio Analytics — Architecture", fontsize=19, fontweight="bold", color=NAVY)
    ax.text(0.05, 0.905, "Account-grain monitoring, leakage-aware modelling and governed AI", fontsize=9.5, color=GREY)

    boxes = [
        (0.04, "Raw data", "5 CSVs: customer, loan, bureau, repayment, collections\npandas read_csv · raw files retained"),
        (0.23, "Validation &\ncleaning", "Key/FK checks · category normalization · missing flags/imputation · chronology flags"),
        (0.42, "Feature layer", "14,685-account master · months 1–10 repayment features · customer bureau aggregates\npandas CSV marts"),
        (0.61, "Risk & anomaly\nmodels", "Logistic + LightGBM/HGB · customer-grouped holdout · Isolation Forest rules\nscikit-learn / LightGBM"),
        (0.80, "Streamlit\ndashboard", "Portfolio · account · Ask · anomaly · methodology\nStreamlit + Plotly · session review state"),
    ]
    for x, title, body in boxes:
        _box(
            ax,
            (x, 0.59),
            0.16,
            0.23,
            title,
            body,
            title_size=8.8,
            body_size=7.0,
            body_wrap=25,
        )
    for x in [0.20, 0.39, 0.58, 0.77]:
        _arrow(ax, (x, 0.705), (x + 0.025, 0.705))

    _box(
        ax, (0.08, 0.26), 0.36, 0.21, "AI feature 1 — Ask the Portfolio",
        "Question → OpenAI Responses API plan → allow-list validation → deterministic pandas calculation → answer + provenance. Sent to the LLM: question, schema and permitted values. No raw rows or calculated metrics.",
        color=ORANGE,
        title_size=10.0,
        body_size=7.3,
        body_wrap=52,
    )
    _box(
        ax, (0.56, 0.26), 0.36, 0.21, "AI feature 2 — Anomaly detection",
        "Account features → robust scaling → Isolation Forest + peer rules → explanation and review queue. Analyst records Confirm / Dismiss / Escalate; production replaces session state with an audited datastore.",
        color=RED,
        title_size=10.0,
        body_size=7.3,
        body_wrap=52,
    )
    _arrow(ax, (0.82, 0.59), (0.40, 0.47))
    _arrow(ax, (0.82, 0.59), (0.74, 0.47))

    ax.add_patch(FancyBboxPatch((0.05, 0.08), 0.90, 0.10, boxstyle="round,pad=0.012", facecolor=PALE, edgecolor="none"))
    ax.text(0.07, 0.145, "Security & operations", fontsize=10.5, fontweight="bold", color=NAVY)
    ax.text(
        0.07, 0.105,
        "OPENAI_API_KEY is read only from environment/hosting secrets · no keys in Git · pinned dependencies · data-quality and model metrics persisted · monitor drift, calibration, segment performance, overrides and API failures",
        fontsize=8.2, color=GREY,
    )
    ax.text(0.95, 0.035, "docs/architecture.pdf · one page", fontsize=7.5, color=GREY, ha="right")
    fig.savefig(DOCS_DIR / "architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def solution_overview_pdf():
    account = pd.read_csv(PROCESSED_DIR / "account_scored.csv")
    metrics = json.loads((MODEL_DIR / "model_metrics.json").read_text(encoding="utf-8"))
    test = metrics["test"]
    income = account.groupby("income_band").target_default.mean()
    employment = account.groupby("employment_type").target_default.mean().sort_values(ascending=False)
    loans = account.groupby("loan_type").target_default.mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#F7F9FB")
    ax.text(0.04, 0.95, "Credit Risk Portfolio Analytics — Solution Overview", fontsize=18, fontweight="bold", color=NAVY)
    ax.text(0.04, 0.915, "A governed monitoring workflow for portfolio analysts", fontsize=9.5, color=GREY)

    sections = [
        (
            0.04, "Dashboard workflow",
            "PORTFOLIO\nKPIs, exposure, risk bands, cohort trends and income/employment/region/age/loan cuts. Filters drive every chart.\n\nACCOUNT REVIEW\nSearch an account to see score, risk band, key indicators, twelve-month repayment trajectory and bureau context.\n\nANOMALY REVIEW\nPrioritized explanations with Confirm, Dismiss or Escalate and an analyst note.",
            TEAL,
        ),
        (
            0.36, "Embedded AI controls",
            "ASK THE PORTFOLIO\nThe LLM translates natural language into an allow-listed metric query. Pandas calculates every number; provenance is shown. Unsupported questions are declined.\n\nTrigger: analyst question.\nFalse positive: wrong query mapping.\nFalse negative: valid question declined.\nHuman step: verify figures/provenance.\n\nANOMALY DETECTION\nIsolation Forest plus peer rules. False positives are legitimate unusual behaviour; false negatives resemble normal peers. Analyst disposition is mandatory.",
            ORANGE,
        ),
        (
            0.68, "Model & evidence",
            f"MODEL\nSelected: {metrics['selected_model'].replace('_',' ').title()}\nCustomer-grouped holdout · months 1–10 only\nAUC {test['roc_auc']:.3f} · KS {test['ks']:.3f} · Gini {test['gini']:.3f}\nPrecision {test['precision']:.3f} · Recall {test['recall']:.3f} · F1 {test['f1']:.3f}\nLogistic retained over HGB, LightGBM and the 0.296-KS blend because it has stronger PR-AUC, recall and F1.\n\nTOP PORTFOLIO SIGNALS\n1. Lowest-income quintile default: {income.get('Q1 Lowest', float('nan')):.1%} vs {income.get('Q5 Highest', float('nan')):.1%} highest-income — prioritize affordability monitoring.\n2. {employment.index[0]} default: {employment.iloc[0]:.1%} — target early outreach.\n3. {loans.index[0]} default: {loans.iloc[0]:.1%} — review product controls and pricing.",
            RED,
        ),
    ]
    for x, title, body, color in sections:
        _box(
            ax,
            (x, 0.35),
            0.28,
            0.50,
            title,
            body,
            color=color,
            body_size=7.1,
            body_wrap=42,
        )

    ax.add_patch(FancyBboxPatch((0.04, 0.09), 0.92, 0.19, boxstyle="round,pad=0.014", facecolor=PALE, edgecolor="none"))
    ax.text(0.06, 0.235, "Assumptions and production needs", fontsize=10.5, fontweight="bold", color=NAVY)
    production = (
        "Assumptions: synthetic snapshot; monitoring/classification use case rather than origination underwriting; bureau fields treated as available at scoring; invalid collection chronology excluded from modelling; analyst decisions persist only for the demo session.\n\n"
        "Bank IT/data requirements: governed feature timestamps and outcome horizon; data contracts and lineage; authenticated role-based access; durable review/audit database; model registry and approval; scheduled scoring; encryption and secrets management; consent/retention controls; fairness review; drift, calibration, data-quality and cost monitoring; incident and rollback procedures."
    )
    ax.text(0.06, 0.195, textwrap.fill(production, 165), fontsize=8.2, color=GREY, va="top", linespacing=1.45)
    ax.text(0.96, 0.035, "docs/solution_overview.pdf · one page", fontsize=7.5, color=GREY, ha="right")
    fig.savefig(DOCS_DIR / "solution_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_documents():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    architecture_pdf()
    solution_overview_pdf()
    return [DOCS_DIR / "architecture.pdf", DOCS_DIR / "solution_overview.pdf"]


if __name__ == "__main__":
    for path in generate_documents():
        print(path)

