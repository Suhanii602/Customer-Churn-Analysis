"""
Generates the final deliverable: a CSV with customer ID, churn
probability, risk band, top reason indicators, and a recommended
action queue — run on the held-out test set.
"""

import joblib
import numpy as np
import pandas as pd

from explain import get_transformed_feature_names
import shap


def assign_risk_band(prob):
    if prob >= 0.6:
        return "High"
    elif prob >= 0.3:
        return "Medium"
    else:
        return "Low"


def assign_queue(risk_band):
    return {
        "High": "Priority retention review",
        "Medium": "Service follow-up",
        "Low": "No action",
    }[risk_band]


if __name__ == "__main__":
    model_pipeline = joblib.load("models/logreg_pipeline.pkl")
    X_test, y_test, ids_test = joblib.load("models/test_data.pkl")

    preprocessor = model_pipeline.named_steps["prep"]
    clf = model_pipeline.named_steps["clf"]

    X_test_transformed = preprocessor.transform(X_test)
    feature_names = get_transformed_feature_names(preprocessor)

    # Predictions
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]

    # Per-customer SHAP reasons
    explainer = shap.LinearExplainer(clf, X_test_transformed)
    shap_values = explainer.shap_values(X_test_transformed)

    def top_reasons(idx, n=3):
        row_shap = shap_values[idx]
        top_idx = np.argsort(np.abs(row_shap))[::-1][:n]
        reasons = []
        for i in top_idx:
            direction = "↑" if row_shap[i] > 0 else "↓"
            reasons.append(f"{feature_names[i]} {direction}")
        return "; ".join(reasons)

    output_rows = []
    for i in range(len(X_test)):
        prob = y_prob[i]
        risk_band = assign_risk_band(prob)
        output_rows.append({
            "customer_id": ids_test.iloc[i],
            "churn_probability": round(prob, 3),
            "risk_band": risk_band,
            "top_reason_indicators": top_reasons(i),
            "recommended_queue": assign_queue(risk_band),
            "actual_churned": y_test.iloc[i],  # kept for your own validation, not a real deployment field
        })

    output_df = pd.DataFrame(output_rows)
    output_df = output_df.sort_values("churn_probability", ascending=False)

    output_df.to_csv("outputs/predictions.csv", index=False)

    print(f"Generated predictions for {len(output_df)} customers")
    print(f"\nRisk band distribution:")
    print(output_df["risk_band"].value_counts())
    print(f"\nTop 5 highest-risk customers:")
    print(output_df.head(5).to_string(index=False))
    print(f"\nSaved to outputs/predictions.csv")