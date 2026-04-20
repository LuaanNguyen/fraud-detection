"""
eval_utils.py
--------------
Shared model evaluation utilities used by both baseline and hybrid pipelines.
"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, classification_report,
)


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train model and return evaluation metrics dict."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
    else:
        y_scores = y_pred.astype(float)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    pr_auc    = average_precision_score(y_test, y_scores)

    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  PR-AUC    : {pr_auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legit','Fraud'])}")

    return {
        "Model"    : name,
        "Precision": round(precision, 4),
        "Recall"   : round(recall, 4),
        "F1-Score" : round(f1, 4),
        "PR-AUC"   : round(pr_auc, 4),
        "y_scores" : y_scores,
    }
