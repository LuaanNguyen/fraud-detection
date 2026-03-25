"""
models.py
----------
Baseline supervised ML models for fraud detection.
Logistic Regression, SVM, Random Forest, XGBoost.
Evaluated using Precision, Recall, F1-Score, and PR-AUC.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, classification_report,
    precision_recall_curve, ConfusionMatrixDisplay,
    confusion_matrix
)

from preprocessing import run_preprocessing

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# Output directory for plots
# ---------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------
# 1. Train & Evaluate a single model
# ---------------------------------------------------------------
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train model and return evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Probability scores for PR-AUC
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
    else:
        y_scores = y_pred

    # Metrics
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
# 2. Plot PR curves for all models
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
# 3. Plot comparison bar chart
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
# 4. Plot confusion matrices
# ---------------------------------------------------------------
def plot_confusion_matrices(results, y_test):
    """Plot confusion matrix for each model."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

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
# 5. Run all baseline models
# ---------------------------------------------------------------
def run_baseline_models():
    """Full pipeline: preprocess → train → evaluate → plot."""

    # Load preprocessed data
    print("\nLoading preprocessed data...")
    prep = run_preprocessing(balance_strategy="oversample")

    X_train = prep["X_train"]
    X_test  = prep["X_test"]
    y_train = prep["y_train"]
    y_test  = prep["y_test"]

    # Define models
    models = [
        ("Logistic Regression", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )),
        ("Random Forest", RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=42, n_jobs=-1
        )),
        ("XGBoost", XGBClassifier(
            n_estimators=100, scale_pos_weight=81,
            random_state=42, eval_metric="aucpr",
            verbosity=0
        )),
    ]

    # Train and evaluate each model
    results = []
    for name, model in models:
        res = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(res)

    # Summary table
    summary_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "y_scores"}
        for r in results
    ])

    print(f"\n{'='*60}")
    print("  SUMMARY TABLE")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))

    # Save summary to CSV
    csv_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  [Saved] Results CSV → {csv_path}")

    # Plots
    plot_pr_curves(results, y_test)
    plot_comparison(summary_df)
    plot_confusion_matrices(results, y_test)

    return summary_df


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------
if __name__ == "__main__":
    run_baseline_models()