"""
Splits data, trains baseline (Logistic Regression) and tree-based
(Random Forest) models, and saves both fitted pipelines to disk.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

from preprocessing import build_preprocessor, get_feature_columns

TARGET_COL = "churned"

def load_data():
    df = pd.read_csv("data/processed/customers_features.csv")
    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df[TARGET_COL]
    ids = df["customer_id"]
    return X, y, ids


def split_data(X, y, ids):
    # First split off test set (15%)
    X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(
        X, y, ids, test_size=0.30, stratify=y, random_state=42
    )
    # Split remaining 30% into val/test (15% each of total)
    X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
        X_temp, y_temp, ids_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    return (X_train, y_train, ids_train,
            X_val, y_val, ids_val,
            X_test, y_test, ids_test)


def train_logreg(X_train, y_train):
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_random_forest(X_train, y_train):
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe

def train_lightgbm(X_train, y_train):
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


if __name__ == "__main__":
    X, y, ids = load_data()
    (X_train, y_train, ids_train,
     X_val, y_val, ids_val,
     X_test, y_test, ids_test) = split_data(X, y, ids)

    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"Train churn rate: {y_train.mean():.2%}")
    print(f"Val churn rate:   {y_val.mean():.2%}")
    print(f"Test churn rate:  {y_test.mean():.2%}")

    print("\nTraining Logistic Regression...")
    logreg = train_logreg(X_train, y_train)

    print("Training Random Forest...")
    rf = train_random_forest(X_train, y_train)

    print("Training LightGBM...")
    lgbm = train_lightgbm(X_train, y_train)
    joblib.dump(lgbm, "models/lgbm_pipeline.pkl")

    # Save everything needed for evaluation in the next step
    joblib.dump(logreg, "models/logreg_pipeline.pkl")
    joblib.dump(rf, "models/rf_pipeline.pkl")
    joblib.dump((X_val, y_val, ids_val), "models/val_data.pkl")
    joblib.dump((X_test, y_test, ids_test), "models/test_data.pkl")

    print("\nModels and data splits saved to models/")