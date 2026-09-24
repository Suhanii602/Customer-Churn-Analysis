
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from preprocessing import get_feature_columns


def get_transformed_feature_names(preprocessor):

    num_features = preprocessor.transformers_[0][2]
    cat_encoder = preprocessor.transformers_[1][1].named_steps["onehot"]
    cat_features = preprocessor.transformers_[1][2]
    cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
    return list(num_features) + list(cat_feature_names)


if __name__ == "__main__":
    model_pipeline = joblib.load("models/logreg_pipeline.pkl")
    X_val, y_val, ids_val = joblib.load("models/val_data.pkl")

    preprocessor = model_pipeline.named_steps["prep"]
    clf = model_pipeline.named_steps["clf"]

    X_val_transformed = preprocessor.transform(X_val)
    feature_names = get_transformed_feature_names(preprocessor)

    # ---- Global explainability ----
    explainer = shap.LinearExplainer(clf, X_val_transformed)
    shap_values = explainer.shap_values(X_val_transformed)

    print("Generating global feature importance plot...")
    shap.summary_plot(
        shap_values, X_val_transformed, feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig("outputs/shap_global_importance.png")
    plt.close()
    print("Saved: outputs/shap_global_importance.png")

    # Also print top drivers as plain numbers for the report
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    print("\nTop 10 churn drivers (by mean |SHAP value|):")
    print(importance_df.head(10).to_string(index=False))
    importance_df.to_csv("outputs/feature_importance.csv", index=False)

    # ---- Per-customer explanations (sample) ----
    y_prob = model_pipeline.predict_proba(X_val)[:, 1]

    def top_reasons_for_customer(idx, n=3):
        row_shap = shap_values[idx]
        top_idx = np.argsort(np.abs(row_shap))[::-1][:n]
        reasons = []
        for i in top_idx:
            direction = "increases" if row_shap[i] > 0 else "decreases"
            reasons.append(f"{feature_names[i]} ({direction} risk)")
        return "; ".join(reasons)

    # Pick 5 high-risk customers as examples for the report
    high_risk_idx = np.argsort(y_prob)[::-1][:5]

    print("\nSample high-risk customer explanations:")
    for idx in high_risk_idx:
        cid = ids_val.iloc[idx]
        prob = y_prob[idx]
        reasons = top_reasons_for_customer(idx)
        print(f"{cid}  |  Churn prob: {prob:.2f}  |  Reasons: {reasons}")