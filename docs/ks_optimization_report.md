# KS Optimization and Data-Filtering Report

## Decision

The existing 10-month logistic model remains the production model. The highest
honest test KS obtained was **29.63%**, from a validation-selected blend of 75%
Extra Trees and 25% logistic regression. It did not exceed the requested 30%
threshold, and its PR-AUC and recall were lower than the production model.

No result was manufactured by selecting a lucky split, choosing exclusions from
the target, tuning on test data, or allowing months 11–12 into the early-warning
feature window.

## Comparable results

| Experiment | Selection population | Validation KS | Test KS | Test AUC | Test PR-AUC | Test precision | Test recall | Test F1 | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Production logistic, months 1–10 | All valid account rows | 26.58% | 28.51% | 67.93% | 31.15% | 23.27% | 73.31% | 35.32% | Retain |
| Hard DQ exclusion | Excluded implausible employment tenure; 6.17% of test | 26.52% | 28.88% | 67.90% | 31.88% | 22.34% | 75.11% | 34.44% | Reject: validation KS lower |
| DQ-rule search winner | Excluded multiple future bureau dates; 1.19% of test | 27.38% | 28.78% | 67.82% | 31.22% | 23.49% | 73.56% | 35.60% | Reject: below target |
| All 12 repayment months | No row filtering | 27.88% | 27.86% | 68.10% | 31.42% | 23.20% | 73.09% | 35.22% | Reject: changes interpretation and no gain |
| Validation-selected nonlinear blend | No row filtering | 29.90% | **29.63%** | 67.82% | 30.60% | 23.50% | 70.55% | 35.26% | Challenger only |
| Same blend refit on development data | No row filtering | 29.90% | 29.37% | 67.53% | 30.65% | 22.96% | 68.64% | 34.41% | Reject |

## Outlier-filter comparison

Outlier filtering was fitted on training predictors only and applied unchanged
to validation and test. The target was never used to decide which rows to drop.

| Rule | Total validation exclusion | Validation KS | Validation AUC | Validation precision | Validation recall | Validation F1 |
|---|---:|---:|---:|---:|---:|---:|
| Hard DQ only | 4.01% | **26.52%** | 67.53% | 23.13% | 74.32% | 35.28% |
| Hard DQ + 2.5% Isolation Forest tail | 6.71% | 25.65% | 66.42% | 22.45% | 73.26% | 34.37% |
| Hard DQ + 5% Isolation Forest tail | 8.84% | 25.06% | 65.83% | 21.81% | 70.79% | 33.35% |

The permitted 10% cap was respected. Larger exclusions were not tried because
the observed direction was consistently negative and would shrink the usable
portfolio without evidence of better generalization.

## Interpretation

- KS is a ranking statistic and does not change when the operating probability
  cutoff changes. Threshold tuning can trade precision against recall, but it
  cannot turn a 29.63% KS into a value above 30%.
- The dataset's strongest defensible challenger is only 0.37 percentage points
  below the requested target. This is within ordinary sampling variability, but
  it cannot be reported as “above 30” on the held-out test data.
- Dropping statistically unusual rows reduced separation. The unusual cases are
  part of the credit-risk population and contain useful hard examples rather
  than obvious corruption.
- The most promising path to a stable KS above 30 is new signal: longer bureau
  history, utilization trends, delinquency recency, verified affordability,
  application behaviour, and a precisely time-anchored default outcome.

## Reproducibility

- `models/experiment_outlier_filter.json` contains the row-filter experiment.
- `models/experiment_dq_rules.json` contains the auditable DQ-rule search.
- `models/experiment_model_search.json` contains the 59-candidate search.
- `models/experiment_final_refit.json` contains the frozen-specification refit.
- `models/experiment_12m/model_metrics.json` contains the 12-month diagnostic.
- Validation comparison tables are saved as CSV files beside these JSON files.

