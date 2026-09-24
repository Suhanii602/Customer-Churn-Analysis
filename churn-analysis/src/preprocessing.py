"""
Builds the preprocessing pipeline: imputation, scaling, and encoding.
Returns an unfitted ColumnTransformer plus the feature lists, so the
same pipeline can be fit on train and reused on val/test/new data.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Columns to drop before modeling — identifiers and the target itself
ID_COLS = ["customer_id"]
TARGET_COL = "churned"

NUMERIC_FEATURES = [
    "tenure_months", "monthly_charges", "total_charges",
    "n_complaints_30d", "reopen_count", "open_unresolved_count",
    "n_contacts_30d", "n_repeat_contacts", "escalation_count",
    "first_contact_resolved", "avg_resolution_hours", "max_resolution_hours",
    "sla_breached", "pending_age_days", "csat_score",
    "days_since_last_complaint",
    "contacts_per_tenure_month", "repeat_contact_rate",
    "unresolved_burden", "service_friction_score", "charges_per_tenure",
]

CATEGORICAL_FEATURES = [
    "plan_type", "internet_service", "payment_method",
    "severity", "channel",
]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])

    return preprocessor


def get_feature_columns():
    """Returns the full list of raw feature column names the
    preprocessor expects, in order."""
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


if __name__ == "__main__":
    df = pd.read_csv("data/processed/customers_features.csv")

    preprocessor = build_preprocessor()
    feature_cols = get_feature_columns()

    X = df[feature_cols]
    y = df[TARGET_COL]

    X_transformed = preprocessor.fit_transform(X)

    print(f"Raw feature columns: {len(feature_cols)}")
    print(f"Transformed shape (after one-hot expansion): {X_transformed.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Any missing values in raw features?\n{X.isnull().sum().sum()} total nulls")