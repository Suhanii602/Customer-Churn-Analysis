# Data Dictionary

## Data Sources

- **Real data**: tenure, contract type, monthly/total charges, internet
  service, payment method, and churn label — sourced from the IBM Telco
  Customer Churn dataset (Cognos/extended version, Kaggle).
- **Synthetic data**: all service-ticket, complaint, contact, resolution,
  and CSAT fields. No public dataset provides this granularity, so these
  were generated synthetically, weakly and noisily correlated with the
  real churn label (not a direct function of it) to simulate realistic
  service-issue-driven churn patterns without introducing target leakage.

## Churn Definition

Churn is defined using the `Churn Value` field from the source dataset:
1 = customer has left, 0 = customer remains active, as recorded at the
time of data collection. This is a **snapshot classification**, not a
forward time-to-churn prediction — there are no timestamped
observation/outcome windows in this dataset to support a "predict churn
in N days" setup.

## Prediction Point

All features represent a customer's attributes at the time of data
collection. No feature is derived from information that would only be
known after the churn outcome — this avoids target leakage per the
project brief's requirement.

---

## Raw / Merged Fields (`data/processed/customers_final.csv`)

| Field | Type | Source | Description |
|---|---|---|---|
| `customer_id` | string | Regenerated | Anonymized sequential ID (`CUST-1000`...); does not map back to the source file's identifiers |
| `tenure_months` | int | Real (Telco) | Number of months the customer has been with the company |
| `plan_type` | categorical | Real (Telco) | Contract type: Month-to-month, One year, Two year |
| `monthly_charges` | float | Real (Telco) | Current monthly billing amount |
| `total_charges` | float | Real (Telco) | Cumulative amount billed over the customer's tenure |
| `internet_service` | categorical | Real (Telco) | DSL, Fiber optic, or No internet service |
| `payment_method` | categorical | Real (Telco) | How the customer pays (electronic check, mailed check, bank transfer, credit card) |
| `n_complaints_30d` | int | Synthetic | Number of service complaints logged in the last 30 days |
| `severity` | categorical | Synthetic | Issue severity of the most relevant complaint: Low, Medium, High |
| `reopen_count` | int | Synthetic | Number of times a support ticket was reopened after initial closure |
| `open_unresolved_count` | int | Synthetic | Number of currently open, unresolved issues |
| `n_contacts_30d` | int | Synthetic | Total support contacts (any channel) in the last 30 days |
| `channel` | categorical | Synthetic | Primary contact channel: Call, Email, Chat, Ticket |
| `n_repeat_contacts` | int | Synthetic | Number of repeat contacts regarding the same issue |
| `escalation_count` | int | Synthetic | Number of times an issue was escalated beyond first-line support |
| `first_contact_resolved` | binary | Synthetic | 1 if the customer's issue was resolved on first contact, else 0 |
| `avg_resolution_hours` | float | Synthetic | Average time (hours) to resolve this customer's issues |
| `max_resolution_hours` | float | Synthetic | Longest single resolution time (hours) among this customer's issues |
| `sla_breached` | binary | Synthetic | 1 if average resolution time exceeded the 48-hour SLA threshold |
| `pending_age_days` | float | Synthetic | Age (days) of currently pending/unresolved issues, weighted by count |
| `csat_score` | float (1–5) | Synthetic | Customer satisfaction score from post-contact surveys |
| `days_since_last_complaint` | int | Synthetic | Days elapsed since the customer's most recent complaint |
| `churned` | binary (target) | Real (Telco) | 1 = customer churned, 0 = active. See Churn Definition above |

## Derived Features (`data/processed/customers_features.csv`)

Added on top of the fields above by `src/features.py`:

| Field | Type | Description |
|---|---|---|
| `contacts_per_tenure_month` | float | `n_contacts_30d / (tenure_months + 1)` — contact intensity relative to how long they've been a customer |
| `repeat_contact_rate` | float | `n_repeat_contacts / (n_contacts_30d + 1)` — what share of contact volume is repeat vs. new issues |
| `unresolved_burden` | float | `open_unresolved_count × pending_age_days` — combines volume and staleness of unresolved issues |
| `service_friction_score` | float | Weighted sum: complaints + 1.5×reopens + 2×escalations + (1 − first_contact_resolved) — single summary friction indicator |
| `charges_per_tenure` | float | `monthly_charges / (tenure_months + 1)` — crude "value at risk" signal |

## Output Fields (`outputs/predictions.csv`)

| Field | Type | Description |
|---|---|---|
| `customer_id` | string | Anonymized customer identifier |
| `churn_probability` | float (0–1) | Model-predicted probability of churn |
| `risk_band` | categorical | Low (<0.3), Medium (0.3–0.6), High (≥0.6) |
| `top_reason_indicators` | string | Top 3 SHAP-driven features for this customer's prediction, with direction (↑ increases risk, ↓ decreases risk) |
| `recommended_queue` | categorical | Action routing: "Priority retention review" (High), "Service follow-up" (Medium), "No action" (Low) |
| `actual_churned` | binary | Ground-truth label, retained only for internal validation during development — **not present in a real deployment scenario**, since a live system predicting future churn would not have this available |

## Privacy Notes

- `customer_id` is a regenerated sequential identifier with no link back
  to the source dataset's own IDs.
- No phone numbers, email addresses, payment details, or free-text
  complaint content are included anywhere in the pipeline.