# Customer Churn Analysis — Service-Issue-Driven Churn Prediction

An intern-assignment prototype that predicts which customers are at risk
of churning after experiencing service issues, and generates a
prioritized, explainable list for a retention team's follow-up.

## Business Problem

Customers who contact support with unresolved or repeated issues are
more likely to leave. This project builds an early-warning model that
scores each customer's churn risk from their profile, usage, and
service-interaction history, so a retention team can proactively reach
out to the customers most likely to need it — not to replace human
judgment, but to prioritize it.

## Data

- **Real fields** — tenure, contract type, monthly/total charges,
  internet service, payment method, and the churn label — sourced from
  the IBM Telco Customer Churn dataset (Cognos/extended version,
  Kaggle).
- **Synthetic fields** — complaint counts, contact volume, resolution
  time, reopen/escalation counts, and CSAT — generated to simulate
  realistic service-issue-driven churn, since no public dataset
  provides ticket-level detail at this granularity. These are weakly
  and noisily correlated with the real churn label (not a direct
  function of it) to avoid target leakage.

Full field-by-field details and the leakage-avoidance rationale are in
`data_dictionary.md`.

## Churn Definition

Churn is defined using the dataset's `Churn Value` field: 1 = customer
has left, 0 = customer remains active, as recorded at the time of data
collection. This is a **snapshot classification**, not a forward
time-to-churn forecast — there are no timestamped observation/outcome
windows in this dataset to support "predict churn in N days." The model
predicts current churn risk from a customer's present profile, usage,
and recent service-interaction attributes.

## How to Run

```bash
python src/generate_data.py       # builds the dataset (Telco + synthetic support layer)
python src/features.py            # feature engineering
python src/preprocessing.py       # (optional check) preprocessing pipeline
python src/train.py               # trains Logistic Regression, Random Forest, LightGBM
python src/evaluate.py            # evaluation metrics, threshold analysis, model comparison
python src/explain.py             # SHAP global importance + sample customer explanations
python src/generate_output.py     # final predictions.csv
streamlit run app/dashboard.py    # optional interactive dashboard
```

## Models Compared

Three models were trained and evaluated on a held-out validation set,
stratified to preserve the ~26.5% churn rate across train/val/test:

| Model | ROC-AUC | PR-AUC | Recall (churn) | Precision (churn) |
|---|---|---|---|---|
| **Logistic Regression** | **0.886** | **0.736** | 0.85 | 0.56 |
| Random Forest | 0.865 | 0.675 | 0.73 | 0.61 |
| LightGBM | 0.863 | 0.685 | 0.74 | 0.60 |

**Logistic Regression was selected as the primary model.** It
outperformed both tree-based alternatives on ROC-AUC, PR-AUC, and
recall for the churn class. This is a real result worth explaining
rather than a surprise to dismiss: the churn signal in this dataset is
dominated by largely linear, monotonic relationships (tenure, contract
type, charges), which logistic regression captures cleanly — the extra
capacity tree models offer for non-linear interactions doesn't have
much to work with here. Random Forest and LightGBM are retained as
comparison models per the assignment's requirement to evaluate at least
two model families.

## Decision Threshold

**Action threshold: 0.5** (85% recall, 56% precision on validation).

This was chosen as a business trade-off, not because it maximizes any
single metric. At lower thresholds (e.g. 0.3), recall rises to 94% but
the flagged list balloons to 56% of the entire customer base — no
longer a "priority" list a retention team could meaningfully act on.
At higher thresholds (e.g. 0.7), the list shrinks to a genuinely
prioritized 23% of customers, but the model misses nearly 40% of actual
churners, undermining the point of an early-warning system. 0.5 sits at
the point where the majority of churners are still caught while the
flagged group remains meaningfully smaller than "almost half the
customer base" — appropriate for a system meant to route customers to
a team doing real, individual follow-up rather than a mass outreach
campaign.

Separately, a **top-decile lift analysis** on Random Forest showed the
top 10% of predicted-risk customers had a 78.1% actual churn rate
against a 26.5% baseline — a 2.95x lift, confirming the model
meaningfully concentrates risk rather than just reflecting the base
rate.

## Key Churn Drivers (SHAP)

Top drivers by mean absolute SHAP value on the validation set:

1. `tenure_months` — the single strongest signal; new customers churn
   far more than long-tenured ones
2. `plan_type` (Two year / Month-to-month) — contract length is a major
   driver in both directions
3. `internet_service` (Fiber optic / No service)
4. `total_charges`, `charges_per_tenure`
5. `avg_resolution_hours`, `n_contacts_30d`, `reopen_count` — the
   service-issue signals this project specifically targets, present
   but appropriately smaller than the core account/contract fields

Global feature importance and per-customer sample explanations are in
`outputs/feature_importance.csv` and printed by `src/explain.py`. Not
every high-risk customer's top reason is service-related — some are
driven primarily by tenure or contract terms independent of any support
issue. This is reported honestly rather than only highlighting the
complaint-driven examples, consistent with the assignment's requirement
not to present correlation as proof of causation.

## Deliverables

- [x] README (this file)
- [x] Data dictionary + documented churn definition (`data_dictionary.md`)
- [x] EDA notebook with written observations (`notebooks/01_eda.ipynb`)
- [x] Feature engineering + preprocessing code (`src/features.py`, `src/preprocessing.py`)
- [x] Model training/comparison — 3 models (`src/train.py`, `src/evaluate.py`)
- [x] Evaluation report — metrics, confusion matrix, threshold rationale (above)
- [x] Feature importance + sample customer-level explanations (`src/explain.py`)
- [x] Predictions CSV — anonymized ID, probability, risk band, reasons (`outputs/predictions.csv`)
- [x] Interactive dashboard (`app/dashboard.py`)
- [ ] Final presentation

## Limitations

This is a prioritization tool, not a guarantee that any individual
customer will leave, and not proof that a specific service issue caused
churn. All high-risk predictions require human review before contact —
the model surfaces candidates for outreach, it does not make retention
decisions. The churn label used here is a snapshot of status at data
collection time, not a forward-looking forecast window. The
service-issue fields in this dataset are synthetic, generated to
simulate realistic patterns since no public dataset provided real
ticket-level detail at the needed granularity — results should be
read as a methodology demonstration, not as findings about any real
customer base.