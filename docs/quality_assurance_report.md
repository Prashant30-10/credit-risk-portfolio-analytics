# Deliverable Quality Assurance Report

Review date: 31 October 2026  
Scope: source code, five raw CSVs, processed datasets, model artifacts, dashboard, AI controls, anomaly workflow, required PDFs, tests, security hygiene and deployment package.

## Executive assessment

The submission is **assignment-ready and technically coherent**, but it should be described as a portfolio-monitoring prototype rather than a bank-production credit decision system. The strongest parts are the reproducible account-grain data pipeline, leakage controls, explainable dashboard, deterministic grounding of the LLM feature and unusually candid model comparison. The principal limitations are the modest separation available in the supplied data, the lack of probability calibration, a large anomaly review queue, session-only analyst decisions and the absence of a live end-to-end LLM test in this environment.

| Area | Score | Assessment |
|---|---:|---|
| Data engineering and controls | 8.5/10 | Correct account grain, protected joins, documented flags and reproducible outputs. Source chronology defects remain material and require upstream remediation. |
| Modelling discipline | 8.0/10 | Customer-grouped holdout, frozen threshold, multiple challengers and honest test reporting. Discrimination is moderate and the score is not a calibrated probability of default. |
| Dashboard and analyst usability | 8.5/10 | All required views, filters, drill-downs, insights and methodology render correctly. Production access control and durable review storage are not implemented. |
| LLM feature design | 8.0/10 | Strong restricted-query architecture: the model plans, application code calculates, unsupported requests are refused, and provenance is shown. A live hosted call still needs verification with a secret. |
| Anomaly workflow | 7.5/10 | Hybrid multivariate/rule approach, plain-language reasons and human dispositions are present. The 12.7% queue is too large without review-capacity tuning. |
| Documentation and reproducibility | 9.0/10 | Requirements map, pinned environment, reproducible build, two one-page PDFs and experiment evidence are supplied. |
| Security and deployment hygiene | 8.5/10 | No key-like token was found; secret loading and safe disabled state are correct; Docker deployment assets are present. Authentication, authorization, persistence and operational controls remain host/bank responsibilities. |
| Assignment readiness | **9.0/10** | Complete except for the externally hosted URL, public repository link and recorded demo. |
| Bank-production readiness | **6.5/10** | Suitable for controlled demonstration and further validation, not for automated lending or collections decisions. |

## Evidence reviewed

- Five source tables: 10,000 customers, 14,685 loan accounts, 22,979 bureau rows, 176,220 repayment rows and 8,635 collection events.
- One-row-per-account analytical and scored datasets: 14,685 unique account IDs with complete target and score fields.
- Production model artifact, metrics, feature importance, segment metrics and retained experiment outputs.
- Dashboard pages: Portfolio, Account Review, Ask the Portfolio, Anomaly Review and Methodology.
- Six automated checks covering deterministic portfolio calculations, rejection of invalid filters, account-grain contracts, model-output completeness, anomaly explanations and exact one-page PDF requirements.
- Static secret scan, Python compilation, local Streamlit rendering and visual review of both PDFs.

## Data quality

### What is strong

- Raw tables are retained and one-to-many sources are aggregated before joining, preventing exposure and default counts from being multiplied.
- Account ID uniqueness, binary target integrity and a plausible overall default rate (16.10%) are tested.
- Category normalization, missingness flags, median imputation and chronology checks are visible in code and documentation.
- Collections are excluded from model inputs, avoiding a high-risk post-event leakage path.
- Months 11–12 are excluded from the selected feature window and customer grouping prevents the same borrower from crossing train, validation and test partitions.

### Material issues in the supplied data

The quality summary identifies 600 standardized gender labels, 510 implausible employment tenures, 7,293 customer-since dates after loan origination, 3,875 collection events before origination, 1,958 future bureau close dates and 1,008 customers without bureau records. These are not cosmetic: the chronology counts are large enough that a production bank pipeline should quarantine affected records or obtain corrected source fields.

The project flags these conditions and excludes invalid collection timing from time-based analysis. It does **not** blindly remove up to 10% of accounts merely to improve a score. Controlled experiments showed that hard data-quality exclusion and 2.5%/5% outlier removal did not improve validation KS or F1. Retaining the cleanest defensible full-account model is therefore the sounder decision.

## Model quality and model choice

The production logistic model was selected on validation data using the highest F1 subject to a 70% recall floor, with PR-AUC as the tie-breaker. Its untouched test results are:

| Metric | Test result | Interpretation |
|---|---:|---|
| ROC-AUC | 0.6793 | Moderate rank ordering; useful but not strong enough for autonomous decisions. |
| PR-AUC | 0.3115 | 1.95× the 16.10% event-rate baseline. |
| KS | 0.2851 | 28.51%; just below the requested 30% target. |
| Gini | 0.3586 | Consistent with the AUC result. |
| Precision | 0.2327 | About 23 defaults per 100 threshold-positive accounts. False-positive workload is high. |
| Recall | 0.7331 | Captures about 73% of defaults, aligned with an early-warning use case. |
| F1 | 0.3532 | Best balanced threshold result among the principal validation-selected candidates. |
| Brier score | 0.2247 | The class-weighted score is not sufficiently calibrated to be presented as a probability of default. |
| Top-20% precision | 0.3068 | The highest-risk fifth contains defaults at 1.92× the portfolio rate. |

The best nonlinear blend reached 29.63% test KS, but had lower PR-AUC (0.3060), recall (0.7055) and F1 (0.3526). A later refit reached 29.37% KS and lower F1 (0.3441). The requested 30% KS was not achieved by any defensible, leakage-controlled experiment. Retaining logistic regression is justified by the agreed selection rule, interpretability and stronger operating-point performance—not by an assumption that a baseline is always preferable.

Before bank production, calibrate the selected score on a time-based out-of-sample period, establish economic cut-offs from review capacity and loss/cost assumptions, run fairness and stability testing, and validate on a genuinely later outcome cohort. Age and gender should not be used for automated adverse action without legal, policy and model-risk approval; in this prototype they are primarily monitoring dimensions.

## Dashboard quality

The dashboard loaded successfully and reconciled to 14,685 accounts, $601.1M original exposure, 16.1% observed default rate and 20.0% high-risk-band share. Portfolio filters feed the charts and findings; account search exposes score, risk band, key indicators, repayment trajectory and bureau context; the anomaly page displays explanations and analyst controls; the methodology page reconciles the published model metrics and selection rationale.

The five decision-oriented insights render with exact portfolio values. The visual hierarchy is clear and the light theme is legible. Both required PDFs now render as clean, non-overlapping, one-page documents.

Production gaps: analyst decisions live only in Streamlit session state; there is no authentication, role-based access, row-level entitlement, durable audit log, database transaction handling or accessibility test suite. Those omissions are acceptable for the assignment demo but not for a bank deployment.

## AI and anomaly controls

The Ask feature has an appropriate safety boundary. The OpenAI model is restricted to a typed schema of approved metrics, grouping dimensions, exact categorical values and a maximum result size. It receives the question and allowed schema—not raw account rows—and application code performs the calculation. Unsupported questions are declined and provenance is returned. The page also fails safely when `OPENAI_API_KEY` is absent.

The remaining validation step is a live call using a host-side secret. That secret was intentionally not available to the review environment, so API authentication, quota and model-access behavior have not been asserted here. Add tests that mock valid, refused, malformed and API-failure responses, then perform one supported and one unsupported hosted smoke test.

The anomaly workflow combines robust scaling, a 300-tree Isolation Forest, peer-relative robust z-scores and transparent repayment/bureau rules. It flags 1,868 accounts (12.7%); 54.2% of flags are in the high-risk band. Explanations and Confirm/Dismiss/Escalate controls make it auditable for a demo, but the queue should be tuned to actual analyst capacity and evaluated with disposition outcomes before operational use.

## Security, reproducibility and packaging

- No key-like value was found. `.env.example` is blank and the app reads `OPENAI_API_KEY` only from Streamlit secrets or the environment.
- Dependencies are version-pinned and the build has deterministic random states.
- The code compiled and all six automated checks passed.
- Docker and Streamlit settings are included, and the app binds to the hosting port without embedding secrets.
- Large alternative-score CSVs and a refit-only model are experimental, not runtime dependencies. The reviewed deployment package excludes them while preserving the experiment metrics and reproducible scripts.

For a real environment, add dependency-vulnerability and license scans, centralized secrets, identity and access management, encryption and retention controls, structured logs, service-level monitoring, database-backed review records, model registry approval, drift/calibration alerts, incident response and rollback procedures.

## Final disposition

**Approve for assignment submission and controlled demonstration after hosting.**  
**Do not approve for automated credit or collections decisions without independent model validation, temporal back-testing, calibration, fairness/legal review, access controls and operational persistence.**

