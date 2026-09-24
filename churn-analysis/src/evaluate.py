import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve, RocCurveDisplay
)

def evaluate_model(model, X_val, y_val, model_name, threshold=0.5):
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\n{'='*50}")
    print(f"{model_name}  (threshold = {threshold})")
    print(f"{'='*50}")
    print(classification_report(y_val, y_pred, target_names=["No Churn", "Churn"]))

    roc_auc = roc_auc_score(y_val, y_prob)
    pr_auc = average_precision_score(y_val, y_prob)
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"PR-AUC:  {pr_auc:.3f}")

    cm = confusion_matrix(y_val, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    print("  (rows = actual [No Churn, Churn], cols = predicted [No Churn, Churn])")

    return y_prob, roc_auc, pr_auc


def lift_at_top_decile(y_val, y_prob):
    df = pd.DataFrame({"y": y_val.values, "prob": y_prob})
    df = df.sort_values("prob", ascending=False)
    top_decile_n = int(len(df) * 0.10)
    top_decile_churn_rate = df.head(top_decile_n)["y"].mean()
    overall_churn_rate = df["y"].mean()
    lift = top_decile_churn_rate / overall_churn_rate
    print(f"\nTop-decile churn rate: {top_decile_churn_rate:.2%}")
    print(f"Overall churn rate:    {overall_churn_rate:.2%}")
    print(f"Lift: {lift:.2f}x")
    return lift


def plot_threshold_tradeoff(y_val, y_prob, model_name):
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions[:-1], label="Precision")
    plt.plot(thresholds, recalls[:-1], label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Precision/Recall vs Threshold — {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"outputs/threshold_tradeoff_{model_name.replace(' ', '_')}.png")
    plt.show()

def threshold_report(y_val, y_prob, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """
    Shows precision/recall/flagged-customer-count at several thresholds,
    so the choice can be framed as a business trade-off rather than a
    default or an accuracy-maximizing pick.
    """
    print("\n" + "="*60)
    print("THRESHOLD COMPARISON")
    print("="*60)
    print(f"{'Threshold':<10}{'Precision':<12}{'Recall':<10}{'Flagged':<10}{'True Churners Caught'}")

    from sklearn.metrics import precision_score, recall_score

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        flagged = y_pred.sum()
        caught = ((y_pred == 1) & (y_val == 1)).sum()
        print(f"{t:<10}{prec:<12.2f}{rec:<10.2f}{flagged:<10}{caught}")



if __name__ == "__main__":
    logreg = joblib.load("models/logreg_pipeline.pkl")
    rf = joblib.load("models/rf_pipeline.pkl")
    X_val, y_val, ids_val = joblib.load("models/val_data.pkl")

    prob_logreg, roc_lr, pr_lr = evaluate_model(logreg, X_val, y_val, "Logistic Regression")
    prob_rf, roc_rf, pr_rf = evaluate_model(rf, X_val, y_val, "Random Forest")

    print("\n" + "="*50)
    print("LIFT ANALYSIS (Random Forest)")
    print("="*50)
    lift_at_top_decile(y_val, prob_rf)

    plot_threshold_tradeoff(y_val, prob_rf, "Random Forest")


    joblib.dump({"logreg": prob_logreg, "rf": prob_rf}, "models/val_probabilities.pkl")
    print("\n" + "="*60)
    print("THRESHOLD SELECTION — Logistic Regression")
    print("="*60)
    threshold_report(y_val, prob_logreg)

    lgbm = joblib.load("models/lgbm_pipeline.pkl")
    prob_lgbm, roc_lgbm, pr_lgbm = evaluate_model(lgbm, X_val, y_val, "LightGBM")

    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Model':<25}{'ROC-AUC':<12}{'PR-AUC'}")
    print(f"{'Logistic Regression':<25}{roc_lr:<12.3f}{pr_lr:.3f}")
    print(f"{'Random Forest':<25}{roc_rf:<12.3f}{pr_rf:.3f}")
    print(f"{'LightGBM':<25}{roc_lgbm:<12.3f}{pr_lgbm:.3f}")