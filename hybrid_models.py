"""
hybrid_models.py
-----------------
Combines graph-derived features with tabular data.
Retrains ML models on the enriched feature set.
Compares performance against baseline models.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, classification_report,
    precision_recall_curve
)

from preprocessing import load_data
from graph import FraudGraph

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RANDOM_STATE = 42


# ---------------------------------------------------------------
# 1. Build enriched feature set
# ---------------------------------------------------------------
def build_hybrid_features(df):
    """Merge tabular + graph features into one dataframe."""
    print("\n[INFO] Building hybrid feature set...")

    # Encode categoricals
    le = LabelEncoder()
    for col in ["age", "gender", "category"]:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # Scale amount
    scaler = StandardScaler()
    df["amount"] = scaler.fit_transform(df[["amount"]])

    # Extract graph features
    fg = FraudGraph()
    df = fg.extract_degree_centrality(df)
    df = fg.extract_pagerank(df)
    df = fg.extract_merchant_fraud_rate(df)
    fg.close()

    # Drop ID columns
    drop_cols = ["customer", "merchant", "step", "zipcodeOri", "zipMerchant"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    print(f"  Feature columns: {[c for c in df.columns if c != 'fraud']}")
    print(f"  Total features : {len(df.columns) - 1}")

    return df


# ---------------------------------------------------------------
# 2. Evaluate a model
# ---------------------------------------------------------------
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train and evaluate a model."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
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
# 3. Plot baseline vs hybrid comparison
# ---------------------------------------------------------------
def plot_comparison(baseline_df, hybrid_df):
    """Side by side comparison of baseline vs hybrid models."""
    metrics = ["Precision", "Recall", "F1-Score", "PR-AUC"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Baseline vs Graph-Enhanced Models", fontsize=16, fontweight="bold")

    models = baseline_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    for i, metric in enumerate(metrics):
        ax = axes[i]
        baseline_vals = baseline_df[metric].tolist()
        hybrid_vals   = hybrid_df[metric].tolist()

        ax.bar(x - width/2, baseline_vals, width,
               label="Baseline", color="#8C1D40", alpha=0.85)
        ax.bar(x + width/2, hybrid_vals, width,
               label="Graph-Enhanced", color="#FFC627", alpha=0.85)

        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "baseline_vs_hybrid.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  [Saved] Comparison chart → {path}")


# ---------------------------------------------------------------
# 4. Plot PR curves comparison
# ---------------------------------------------------------------
def plot_pr_comparison(baseline_results, hybrid_results, y_test):
    """PR curves for baseline vs hybrid."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#8C1D40", "#FFC627", "#1A6B3C"]

    for ax, results, title in zip(
        axes,
        [baseline_results, hybrid_results],
        ["Baseline Models", "Graph-Enhanced Models"]
    ):
        for i, res in enumerate(results):
            p, r, _ = precision_recall_curve(y_test, res["y_scores"])
            ax.plot(r, p, label=f"{res['Model']} ({res['PR-AUC']:.3f})",
                    color=colors[i % len(colors)], linewidth=2)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("PR Curves: Baseline vs Graph-Enhanced", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "pr_curves_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Saved] PR curves comparison → {path}")


# ---------------------------------------------------------------
# 5. Full hybrid pipeline
# ---------------------------------------------------------------
def run_hybrid_models():
    # Load and enrich data
    df = load_data()
    df = build_hybrid_features(df)

    # Features and labels
    X = df.drop(columns=["fraud"])
    y = df["fraud"]

    # Balance with oversampling
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_bal, y_bal = ros.fit_resample(X, y)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2,
        stratify=y_bal, random_state=RANDOM_STATE
    )

    print(f"\n  Training set : {X_train.shape[0]:,} samples")
    print(f"  Test set     : {X_test.shape[0]:,} samples")
    print(f"  Features     : {X_train.shape[1]}")

    # Define models
    models = [
        ("Logistic Regression", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        )),
        ("Random Forest", RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        )),
        ("XGBoost", XGBClassifier(
            n_estimators=100, scale_pos_weight=81,
            random_state=RANDOM_STATE, eval_metric="aucpr",
            verbosity=0
        )),
    ]

    # Train and evaluate
    hybrid_results = []
    for name, model in models:
        res = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        hybrid_results.append(res)

    # Summary
    hybrid_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "y_scores"}
        for r in hybrid_results
    ])

    print(f"\n{'='*60}")
    print("  HYBRID MODEL SUMMARY")
    print(f"{'='*60}")
    print(hybrid_df.to_string(index=False))

    # Save results
    csv_path = os.path.join(RESULTS_DIR, "hybrid_results.csv")
    hybrid_df.to_csv(csv_path, index=False)
    print(f"\n  [Saved] Results → {csv_path}")

    # Load baseline for comparison
    baseline_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
    if os.path.exists(baseline_path):
        baseline_df = pd.read_csv(baseline_path)
        # Remove SVM if present
        baseline_df = baseline_df[baseline_df["Model"] != "SVM"].reset_index(drop=True)

        print(f"\n{'='*60}")
        print("  BASELINE vs HYBRID COMPARISON")
        print(f"{'='*60}")
        print("\nBaseline:")
        print(baseline_df.to_string(index=False))
        print("\nHybrid (Graph-Enhanced):")
        print(hybrid_df.to_string(index=False))

        plot_comparison(baseline_df, hybrid_df)
        plot_pr_comparison(
            [r for r in hybrid_results],
            [r for r in hybrid_results],
            y_test
        )

    return hybrid_df


if __name__ == "__main__":
    run_hybrid_models()