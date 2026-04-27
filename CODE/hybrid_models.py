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
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import RandomOverSampler

from preprocessing import load_data
from graph import FraudGraph
from eval_utils import evaluate_model

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "EVALUATIONS")
os.makedirs(RESULTS_DIR, exist_ok=True)
RANDOM_STATE = 42


# ---------------------------------------------------------------
# 1. Encode and scale tabular features (leak-safe)
# ---------------------------------------------------------------
def encode_and_scale(df):
    """Label-encode categoricals and standard-scale numerics.

    Returns the transformed DataFrame plus fitted encoders/scaler so
    the same mapping can be applied to held-out data.
    """
    encoders: dict[str, LabelEncoder] = {}
    for col in ["age", "gender", "category"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    scaler = StandardScaler()
    df["amount"] = scaler.fit_transform(df[["amount"]])

    return df, encoders, scaler


# ---------------------------------------------------------------
# 2. Add graph features (degree, PageRank, betweenness, community)
#    Merchant fraud rate is added *after* the train/test split
#    to avoid data leakage.
# ---------------------------------------------------------------
def add_graph_features(df):
    """Attach degree centrality, PageRank, betweenness, and community columns."""
    fg = FraudGraph()
    df = fg.extract_degree_centrality(df)
    df = fg.extract_pagerank(df)
    df = fg.extract_betweenness_centrality(df)
    df = fg.extract_community_ids(df)
    fg.close()
    return df


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
    df = load_data()
    df, encoders, scaler = encode_and_scale(df)
    df = add_graph_features(df)

    drop_cols = ["customer", "merchant", "step", "zipcodeOri", "zipMerchant"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    # Keep customer/merchant IDs around for the fraud-rate join, drop later
    id_cols_present = [c for c in ["customer", "merchant"] if c in df.columns]

    X = df.drop(columns=[c for c in drop_cols if c not in id_cols_present] + ["fraud"])
    y = df["fraud"]

    # Balance with oversampling
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_bal, y_bal = ros.fit_resample(X, y)

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2,
        stratify=y_bal, random_state=RANDOM_STATE
    )

    # Compute merchant_fraud_rate from TRAINING data only, then map to test
    fg = FraudGraph()
    train_ref = pd.DataFrame({"merchant": X_train["merchant"], "fraud": y_train})
    X_train = fg.extract_merchant_fraud_rate(X_train, reference_df=train_ref)
    X_test = fg.extract_merchant_fraud_rate(X_test, reference_df=train_ref)
    fg.close()

    # Now drop the ID columns that were kept for the join
    for col in id_cols_present:
        if col in X_train.columns:
            X_train = X_train.drop(columns=[col])
            X_test = X_test.drop(columns=[col])

    print(f"\n  Training set : {X_train.shape[0]:,} samples")
    print(f"  Test set     : {X_test.shape[0]:,} samples")
    print(f"  Features     : {list(X_train.columns)}")

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

    hybrid_results = []
    for name, model in models:
        res = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        hybrid_results.append(res)

    hybrid_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "y_scores"}
        for r in hybrid_results
    ])

    print(f"\n{'='*60}")
    print("  HYBRID MODEL SUMMARY")
    print(f"{'='*60}")
    print(hybrid_df.to_string(index=False))

    csv_path = os.path.join(RESULTS_DIR, "hybrid_results.csv")
    hybrid_df.to_csv(csv_path, index=False)
    print(f"\n  [Saved] Results → {csv_path}")

    # Load baseline for comparison
    baseline_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
    if os.path.exists(baseline_path):
        baseline_df = pd.read_csv(baseline_path)

        print(f"\n{'='*60}")
        print("  BASELINE vs HYBRID COMPARISON")
        print(f"{'='*60}")
        print("\nBaseline:")
        print(baseline_df.to_string(index=False))
        print("\nHybrid (Graph-Enhanced):")
        print(hybrid_df.to_string(index=False))

        plot_comparison(baseline_df, hybrid_df)

        # Re-run baseline models to get y_scores for PR curve comparison
        from models import run_baseline_models as _run_baseline
        baseline_results_obj = _get_baseline_results_for_pr(X_train, X_test, y_train, y_test)
        if baseline_results_obj:
            plot_pr_comparison(baseline_results_obj, hybrid_results, y_test)

    return hybrid_df


def _get_baseline_results_for_pr(X_train, X_test, y_train, y_test):
    """Quick retrain of baseline models (without graph features) for PR curves.

    Returns list of result dicts with y_scores, or empty list on failure.
    """
    try:
        from preprocessing import run_preprocessing
        prep = run_preprocessing(balance_strategy="oversample")
        base_models = [
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
        results = []
        for name, model in base_models:
            res = evaluate_model(
                name, model,
                prep["X_train"], prep["X_test"],
                prep["y_train"], prep["y_test"],
            )
            results.append(res)
        return results
    except Exception as e:
        print(f"  [WARNING] Could not build baseline PR curves: {e}")
        return []


if __name__ == "__main__":
    run_hybrid_models()
