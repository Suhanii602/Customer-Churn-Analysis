"""
Builds the final churn dataset by combining the real IBM Telco base
with a synthetic service-issue/support-contact layer.

IMPORTANT: the synthetic features are only weakly, realistically
correlated with churn (small coefficients + heavy independent noise),
not deterministic functions of it. A too-strong correlation here would
be target leakage — the model would just be reverse-engineering the
label instead of learning a genuine pattern.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ---- Load real base data ----
telco = pd.read_excel("data/raw/telco_base.xlsx")

telco["Total Charges"] = pd.to_numeric(telco["Total Charges"], errors="coerce")
telco["Total Charges"] = telco["Total Charges"].fillna(telco["Total Charges"].median())
telco["churned"] = telco["Churn Value"].astype(int)

N = len(telco)
customer_id = [f"CUST-{1000+i}" for i in range(N)]

base = pd.DataFrame({
    "customer_id": customer_id,
    "tenure_months": telco["Tenure Months"],
    "plan_type": telco["Contract"],
    "monthly_charges": telco["Monthly Charges"],
    "total_charges": telco["Total Charges"],
    "internet_service": telco["Internet Service"],
    "payment_method": telco["Payment Method"],
    "churned": telco["churned"],
})

churn = base["churned"].values

# ---- Independent per-customer noise, NOT derived from churn ----
# This represents natural variation in service experience that isn't
# fully explained by whether someone eventually churns.
service_quality_noise = np.random.normal(0, 1, N)

# ---- Synthetic support-ticket layer ----
# Small coefficients on `churn` = realistic partial signal.
# Large base rates + noise = most of the variance is NOT from churn,
# same as real support data would look.

n_complaints_30d = np.random.poisson(0.9 + 0.35 * churn, N)
severity = np.random.choice(["Low", "Medium", "High"], N, p=[0.5, 0.35, 0.15])
reopen_count = np.random.poisson(0.3 + 0.15 * churn, N)
open_unresolved_count = np.random.poisson(0.3 + 0.15 * churn, N)

n_contacts_30d = n_complaints_30d + np.random.poisson(0.4 + 0.1 * churn, N)
channel = np.random.choice(["Call", "Email", "Chat", "Ticket"], N)
n_repeat_contacts = np.random.poisson(0.2 + 0.1 * churn, N)
escalation_count = np.random.poisson(0.15 + 0.08 * churn, N)

first_contact_resolved = np.random.binomial(
    1, np.clip(0.62 - 0.08 * churn, 0.1, 0.9), N
)
avg_resolution_hours = np.random.exponential(18 + 5 * churn, N)
max_resolution_hours = avg_resolution_hours + np.random.exponential(12, N)
sla_breached = (avg_resolution_hours > 48).astype(int)
pending_age_days = (np.random.exponential(1.8, N) * open_unresolved_count).round(1)

csat_score = np.clip(
    np.random.normal(3.9, 0.9, N) - 0.15 * n_complaints_30d - 0.1 * churn, 1, 5
).round(1)

days_since_last_complaint = np.where(
    n_complaints_30d > 0, np.random.randint(1, 30, N), np.random.randint(30, 180, N)
)

support = pd.DataFrame({
    "n_complaints_30d": n_complaints_30d,
    "severity": severity,
    "reopen_count": reopen_count,
    "open_unresolved_count": open_unresolved_count,
    "n_contacts_30d": n_contacts_30d,
    "channel": channel,
    "n_repeat_contacts": n_repeat_contacts,
    "escalation_count": escalation_count,
    "first_contact_resolved": first_contact_resolved,
    "avg_resolution_hours": avg_resolution_hours.round(1),
    "max_resolution_hours": max_resolution_hours.round(1),
    "sla_breached": sla_breached,
    "pending_age_days": pending_age_days,
    "csat_score": csat_score,
    "days_since_last_complaint": days_since_last_complaint,
})

final = pd.concat([base.drop(columns=["churned"]), support, base[["churned"]]], axis=1)

final.to_csv("data/processed/customers_final.csv", index=False)
print(f"Final dataset: {len(final)} rows, {final.shape[1]} columns")
print(f"Churn rate: {final['churned'].mean():.2%}")

# Quick leakage sanity check right here, before you even get to modeling
numeric_check = final.select_dtypes(include="number").corr()["churned"].sort_values(ascending=False)
print("\nCorrelation with churned (should be mild, nothing near 0.9+):")
print(numeric_check)