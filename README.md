---
title: Credit Risk Portfolio Analytics
emoji: 📊
colorFrom: teal
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Credit Risk Portfolio Analytics

A deployable credit-risk monitoring dashboard built from five synthetic banking tables. It combines account-grain portfolio analytics, a leakage-aware default classifier, grounded natural-language portfolio questions, and a human-reviewed anomaly queue.

## What is included

- Portfolio view: exposure, default rate, risk bands, cohort trends and required demographic/product cuts.
- Account review: model score, key risk indicators, repayment trajectory and bureau context.
- Ask the Portfolio: an OpenAI-powered natural-language interface. The model creates a restricted query plan; application code calculates every number and displays provenance.
- Anomaly review: Isolation Forest plus transparent peer/history rules, explanations and Confirm/Dismiss/Escalate decisions.
- Logistic-regression baseline plus HistGradientBoosting and LightGBM challengers.
- AUC, PR-AUC, KS, Gini, precision, recall, F1, calibration and segment monitoring outputs.
- Two required one-page PDF documents.

## Python

Python 3.11.9 was used for development and testing.

## Run in three commands

```bash
python -m pip install -r requirements.txt
python -m src.build_all
python -m streamlit run app/app.py
```

## OpenAI key

The key is never stored in source control. For local use, set an environment variable named `OPENAI_API_KEY`. Optionally set `OPENAI_MODEL`; the default is `gpt-6-luna`.

For Streamlit Community Cloud, open the app's **Settings → Secrets** and add:

```toml
OPENAI_API_KEY = "your-project-key"
OPENAI_MODEL = "gpt-6-luna"
```

If no key is configured, the Ask page displays a disabled-state message while portfolio, account, anomaly and methodology views continue to work.

## Repository map

| Brief item | Location |
|---|---|
| Dashboard entry point | `app/app.py` |
| Ingestion and quality decisions | `src/data_pipeline.py` |
| Portfolio metric engine | `src/analytics.py` |
| Logistic and ML modelling | `src/modeling.py` |
| Grounded LLM assistant | `src/llm_assistant.py` |
| Anomaly detection | `src/anomaly_detection.py` |
| Reproducible full build | `src/build_all.py` |
| Raw five-table data pack | `data/raw/` |
| Analysis-ready/scored data | `data/processed/` |
| Model artifact and metrics | `models/` |
| Architecture document | `docs/architecture.pdf` |
| Solution overview | `docs/solution_overview.pdf` |
| Automated checks | `tests/` |

## Assignment requirement checklist

| Requirement from the brief | Implementation and status |
|---|---|
| Load and join all five tables | Complete in `src/data_pipeline.py`; raw files are preserved in `data/raw/` and one account-grain analytical dataset is produced. |
| Document joins and data-quality decisions | Complete in the README, methodology view, generated quality summary and architecture/solution PDFs. |
| Profile income, employment, region, age and loan type | Complete in the Portfolio view, including a cross-dimensional employment × loan cut and five finding/“so what” statements. |
| Logistic baseline plus one ML model | Complete: logistic regression, HistGradientBoosting and LightGBM, plus nonlinear/ensemble challengers. |
| Report AUC, KS, Gini and feature importance/SHAP | Complete in the Methodology view and model artifacts; permutation importance is used. |
| Explain production monitoring | Complete in the Methodology view, solution overview and Production monitoring section below. |
| Interactive portfolio and account views | Complete in `app/app.py`, with filters, risk segments, exposure, trends, scores, drivers and repayment/bureau context. |
| Grounded “Ask the Portfolio” LLM feature | Complete with an allow-listed query plan, deterministic calculations, provenance and safe refusal. Requires `OPENAI_API_KEY` on the host. |
| Anomaly detection and analyst decision capture | Complete with Isolation Forest, transparent rules, explanations and Confirm/Dismiss/Escalate session decisions. |
| Trigger, false positive, false negative and human step for both AI features | Complete in dashboard expanders and `docs/solution_overview.pdf`. |
| Exactly two one-page PDFs | Complete: `docs/architecture.pdf` and `docs/solution_overview.pdf`. |
| Required repository structure and reproducibility | Complete: pinned packages, `.env.example`, three-command run instructions, Python version and repository map. |
| Hosted dashboard, GitHub repository and 3–5 minute demo links | External submission steps remain: deploy the app, publish/share the repository and record the demo. Do not submit a ZIP. |

## Data and modelling decisions

- The modelling grain is one row per loan account: 14,685 rows after customer, aggregated bureau and aggregated repayment features are joined.
- Raw one-to-many tables are never joined directly because that would duplicate exposure and outcomes.
- `M/F` gender values are normalized to `Male/Female`.
- Missing values are accompanied by flags and imputed with documented segment medians.
- Implausible employment tenure, customer-since chronology, future bureau closures and pre-origination collections are flagged.
- Pre-origination collections are excluded from time-based analysis, and collections are excluded from model features.
- The production model uses repayment months 1–10; months 11–12 remain outside the feature window. A six-month benchmark is retained for comparison.
- Train, validation and test splits are grouped by customer so no customer crosses partitions.
- The operating threshold is selected on validation data by F1 with a 70% recall floor. The test set remains untouched until final evaluation.

The supplied data has limited separability. Metrics are therefore reported honestly rather than improved through leakage or threshold selection on the test set.

## KS optimization results

Target-independent filtering, data-quality rules, nonlinear scorecards, tree
models, probability blends and a 12-month diagnostic were compared. The best
held-out test KS was **29.63%**; no defensible experiment exceeded 30%. The
production logistic model is retained because it provides stronger PR-AUC and
recall with only a modest KS difference. See
`docs/ks_optimization_report.md` for the complete comparison and reproducible
result files.

## AI controls

### Ask the Portfolio

- Trigger: an analyst submits a natural-language question.
- Grounding: the LLM can select only approved metrics, dimensions and exact categorical filters. Pandas calculates all figures.
- False positive: a question is mapped to the wrong approved query.
- False negative: a valid portfolio question is declined.
- Human step: the analyst verifies the displayed query provenance before acting.
- Data sent externally: question, allowed schema and permitted category values. Raw customer/account rows and calculated portfolio figures are not sent to the LLM.

### Anomaly detection

- Trigger: top multivariate anomaly score or a transparent repayment/bureau rule.
- False positive: unusual but legitimate behaviour.
- False negative: risky behaviour that resembles its peer segment.
- Human step: confirm, dismiss or escalate after reviewing the account.
- Demo persistence: review decisions are kept in the current Streamlit session. Production requires a durable, access-controlled audit database.

## Production monitoring

Monitor data contracts, missingness, score and feature PSI, AUC/KS/Gini, precision/recall/F1 at the approved threshold, calibration, segment performance, risk-band migration, analyst overrides, anomaly dispositions, LLM refusal/query-validation rates, API availability and spend.

## Deployment checklist

1. Push this repository to GitHub without `.env` or `.streamlit/secrets.toml`.
2. Create a Streamlit Community Cloud app using `app/app.py`.
3. Add `OPENAI_API_KEY` and `OPENAI_MODEL` in the host's encrypted secrets settings.
4. Confirm the app opens without login and test one supported and one unsupported Ask question.
5. Keep the app live for at least 14 days after submission.

For the assignment-approved Hugging Face Spaces alternative, create a Docker
Space and upload this repository. The included `Dockerfile` serves Streamlit on
port 7860. Add `OPENAI_API_KEY` as a Space secret and `OPENAI_MODEL=gpt-6-luna`
as a variable or secret. Current Hugging Face account eligibility and pricing
for Docker Spaces are determined by the host.
