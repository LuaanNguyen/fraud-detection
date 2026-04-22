"""
models.py
----------
Baseline supervised ML models for fraud detection.
Logistic Regression, SVM, Random Forest, XGBoost.
Evaluated using Precision, Recall, F1-Score, and PR-AUC.
Includes hyperparameter tuning via RandomizedSearchCV.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, classification_report,
    precision_recall_curve, ConfusionMatrixDisplay,
    confusion_matrix
)

from preprocessing import run_preprocessing

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RANDOM_STATE = 42


# ---------------------------------------------------------------
# 1. Train & Evaluate a single model
# ---------------------------------------------------------------
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train model and return evaluation metrics."""
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
        y_scores = y_pred

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


# ---------------------------------------------------------------
# 2. Hyperparameter Tuning
# ---------------------------------------------------------------
def tune_random_forest(X_train, y_train):
    """Tune Random Forest using RandomizedSearchCV."""
    print("\n[INFO] Tuning Random Forest...")

    param_dist = {
        "n_estimators"      : [100, 200, 300],
        "max_depth"         : [None, 10, 20, 30],
        "min_samples_split" : [2, 5, 10],
        "min_samples_leaf"  : [1, 2, 4],
        "max_features"      : ["sqrt", "log2"],
    }

    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    search = RandomizedSearchCV(
        rf, param_dist,
        n_iter=10,
        scoring="average_precision",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)
    print(f"  Best RF params : {search.best_params_}")
    print(f"  Best RF PR-AUC : {search.best_score_:.4f}")
    return search.best_estimator_


def tune_xgboost(X_train, y_train):
    """Tune XGBoost using RandomizedSearchCV."""
    print("\n[INFO] Tuning XGBoost...")

    param_dist = {
        "n_estimators"  : [100, 200, 300],
        "max_depth"     : [3, 5, 7, 9],
        "learning_rate" : [0.01, 0.05, 0.1, 0.2],
        "subsample"     : [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "scale_pos_weight": [81],
    }

    xgb = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="aucpr",
        verbosity=0
    )

    search = RandomizedSearchCV(
        xgb, param_dist,
        n_iter=10,
        scoring="average_precision",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)
    print(f"  Best XGB params : {search.best_params_}")
    print(f"  Best XGB PR-AUC : {search.best_score_:.4f}")
    return search.best_estimator_


def tune_svm(X_train, y_train):
    """Tune SVM using RandomizedSearchCV."""
    print("\n[INFO] Tuning SVM...")

    param_dist = {
        "C"     : [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
        "gamma" : ["scale", "auto"],
    }

    svm = SVC(
        class_weight="balanced",
        probability=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        svm, param_dist,
        n_iter=6,
        scoring="f1",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)
    print(f"  Best SVM params : {search.best_params_}")
    print(f"  Best SVM F1     : {search.best_score_:.4f}")
    return search.best_estimator_


# ---------------------------------------------------------------
# 3. Plot PR curves
# ---------------------------------------------------------------
def plot_pr_curves(results, y_test):
    """Plot Precision-Recall curves for all models."""
    plt.figure(figsize=(10, 6))
    colors = ["#8C1D40", "#FFC627", "#1A6B3C", "#1F77B4"]

    for i, res in enumerate(results):
        precision_vals, recall_vals, _ = precision_recall_curve(
            y_test, res["y_scores"]
        )
        plt.plot(
            recall_vals, precision_vals,
            label=f"{res['Model']} (PR-AUC={res['PR-AUC']:.3f})",
            color=colors[i % len(colors)], linewidth=2
        )

    plt.xlabel("Recall", fontsize=13)
    plt.ylabel("Precision", fontsize=13)
    plt.title("Precision-Recall Curves — Baseline Models", fontsize=15, fontweight="bold")
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "pr_curves_baseline.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  [Saved] PR Curves → {path}")


# ---------------------------------------------------------------
# 4. Plot comparison bar chart
# ---------------------------------------------------------------
def plot_comparison(summary_df):
    """Bar chart comparing all models across metrics."""
    metrics = ["Precision", "Recall", "F1-Score", "PR-AUC"]
    x = np.arange(len(summary_df))
    width = 0.2
    colors = ["#8C1D40", "#FFC627", "#1A6B3C", "#1F77B4"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, summary_df[metric], width,
               label=metric, color=colors[i], alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(summary_df["Model"], fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Baseline Model Comparison", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Saved] Comparison Chart → {path}")


# ---------------------------------------------------------------
# 5. Plot confusion matrices
# ---------------------------------------------------------------
def plot_confusion_matrices(results, y_test):
    """Plot confusion matrix for each model."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for i, res in enumerate(results):
        y_pred = (res["y_scores"] >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Legit", "Fraud"]
        )
        disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
        axes[i].set_title(res["Model"], fontsize=13, fontweight="bold")

    plt.suptitle("Confusion Matrices — Baseline Models", fontsize=15, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Saved] Confusion Matrices → {path}")


# ---------------------------------------------------------------
# 6. Run all baseline models
# ---------------------------------------------------------------
def run_baseline_models():
    """Full pipeline: preprocess → tune → train → evaluate → plot."""

    print("\nLoading preprocessed data...")
    prep = run_preprocessing(balance_strategy="oversample")

    X_train = prep["X_train"]
    X_test  = prep["X_test"]
    y_train = prep["y_train"]
    y_test  = prep["y_test"]

    print(f"\n{'='*60}")
    print("  HYPERPARAMETER TUNING")
    print(f"{'='*60}")

    # Tune SVM, RF and XGBoost
    best_svm = tune_svm(X_train, y_train)
    best_rf  = tune_random_forest(X_train, y_train)
    best_xgb = tune_xgboost(X_train, y_train)

    # Define models — LR with defaults, SVM/RF/XGBoost tuned
    models = [
        ("Logistic Regression", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        )),
        ("SVM (Tuned)", best_svm),
        ("Random Forest (Tuned)", best_rf),
        ("XGBoost (Tuned)", best_xgb),
    ]

    # Train and evaluate
    results = []
    for name, model in models:
        res = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(res)

    # Summary
    summary_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "y_scores"}
        for r in results
    ])

    print(f"\n{'='*60}")
    print("  SUMMARY TABLE")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))

    csv_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  [Saved] Results CSV → {csv_path}")

    plot_pr_curves(results, y_test)
    plot_comparison(summary_df)
    plot_confusion_matrices(results, y_test)

    return summary_df


if __name__ == "__main__":
    run_baseline_models()