"""
Feature engineering for the churn model.
Takes the processed customers_final.csv and returns a clean,
model-ready feature matrix + target, with identifiers separated out.
"""

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: the merged customers_final dataframe.
    Output: dataframe with derived features added, ready for the
    preprocessing pipeline. customer_id and churned are kept in
    (they get split off later in train.py).
    """
    feats = df.copy()

    # --- Derived ratio/interaction features ---

    # Contact intensity relative to tenure (long-tenure customers
    # complaining a lot is a stronger signal than new customers doing so)
    feats["contacts_per_tenure_month"] = feats["n_contacts_30d"] / (
        feats["tenure_months"] + 1  # +1 avoids divide-by-zero for tenure=0
    )

    # Repeat-contact rate: how much of their contact volume is repeat vs. new
    feats["repeat_contact_rate"] = feats["n_repeat_contacts"] / (
        feats["n_contacts_30d"] + 1
    )

    # Unresolved burden: open issues combined with how long they've been pending
    feats["unresolved_burden"] = (
        feats["open_unresolved_count"] * feats["pending_age_days"]
    )

    # Service friction score: simple weighted combination of the
    # brief's key service-issue signals, as one summary feature
    feats["service_friction_score"] = (
        feats["n_complaints_30d"] * 1.0
        + feats["reopen_count"] * 1.5
        + feats["escalation_count"] * 2.0
        + (1 - feats["first_contact_resolved"]) * 1.0
    )

    # Charges-per-tenure (crude "value at risk" signal, from real Telco fields)
    feats["charges_per_tenure"] = feats["monthly_charges"] / (
        feats["tenure_months"] + 1
    )

    return feats


if __name__ == "__main__":
    df = pd.read_csv("data/processed/customers_final.csv")
    feats = build_features(df)

    print(f"Original columns: {df.shape[1]}")
    print(f"After feature engineering: {feats.shape[1]}")
    print(feats[[
        "customer_id", "contacts_per_tenure_month", "repeat_contact_rate",
        "unresolved_burden", "service_friction_score", "charges_per_tenure"
    ]].head())

    feats.to_csv("data/processed/customers_features.csv", index=False)
    print("Saved to data/processed/customers_features.csv")