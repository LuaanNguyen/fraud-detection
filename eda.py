"""
eda.py
-------
Exploratory Data Analysis for the BankSim fraud-detection dataset.
Generates distribution plots, correlation heatmaps, and fraud-pattern
analyses.  All figures are saved to the output/eda/ directory.
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

EDA_DIR = os.path.join(os.path.dirname(__file__), "output", "eda")
os.makedirs(EDA_DIR, exist_ok=True)

LABEL_COL = "fraud"


def _save(fig, name: str) -> None:
    path = os.path.join(EDA_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {path}")


# ------------------------------------------------------------------
# 1. Descriptive statistics
# ------------------------------------------------------------------
def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print basic shape, dtypes, and descriptive stats."""
    print(f"\n{'='*60}")
    print("  SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"  Rows    : {df.shape[0]:,}")
    print(f"  Columns : {df.shape[1]}")
    print(f"\n  Dtypes:\n{df.dtypes.to_string()}")
    print(f"\n  Descriptive stats (numeric):")
    print(df.describe().to_string())
    print(f"\n  Unique values per column:")
    for col in df.columns:
        print(f"    {col:<20s}: {df[col].nunique():>8,}")


# ------------------------------------------------------------------
# 2. Class distribution
# ------------------------------------------------------------------
def plot_class_distribution(df: pd.DataFrame) -> None:
    """Bar chart + pie chart of fraud vs legitimate."""
    print(f"\n{'='*60}")
    print("  CLASS DISTRIBUTION")
    print(f"{'='*60}")

    counts = df[LABEL_COL].value_counts().sort_index()
    labels = ["Legitimate", "Fraud"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["#2ecc71", "#e74c3c"]
    ax1.bar(labels, counts.values, color=colors, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(counts.values):
        ax1.text(i, v + v * 0.02, f"{v:,}", ha="center", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Count")
    ax1.set_title("Transaction Counts", fontsize=13, fontweight="bold")
    ax1.ticklabel_format(style="plain", axis="y")

    ax2.pie(
        counts.values, labels=labels, autopct="%1.2f%%",
        colors=colors, startangle=90, textprops={"fontsize": 11},
        wedgeprops={"edgecolor": "black", "linewidth": 0.5},
    )
    ax2.set_title("Class Proportions", fontsize=13, fontweight="bold")

    fig.suptitle("Class Distribution — Fraud vs Legitimate", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, "class_distribution.png")


# ------------------------------------------------------------------
# 3. Transaction amount distributions
# ------------------------------------------------------------------
def plot_amount_distributions(df: pd.DataFrame) -> None:
    """Histograms of transaction amounts by class."""
    print(f"\n{'='*60}")
    print("  AMOUNT DISTRIBUTIONS")
    print(f"{'='*60}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.hist(df["amount"], bins=100, color="#3498db", edgecolor="black", linewidth=0.3)
    ax.set_title("All Transactions", fontsize=13, fontweight="bold")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Frequency")
    ax.set_yscale("log")

    for i, (label, color) in enumerate([(0, "#2ecc71"), (1, "#e74c3c")]):
        ax = axes[i + 1]
        subset = df[df[LABEL_COL] == label]["amount"]
        ax.hist(subset, bins=80, color=color, edgecolor="black", linewidth=0.3)
        tag = "Legitimate" if label == 0 else "Fraud"
        ax.set_title(f"{tag} (n={len(subset):,})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Amount")
        ax.set_ylabel("Frequency")

    fig.suptitle("Transaction Amount Distributions", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, "amount_distributions.png")

    legit = df[df[LABEL_COL] == 0]["amount"]
    fraud = df[df[LABEL_COL] == 1]["amount"]
    print(f"  Legit  — mean: {legit.mean():.2f}, median: {legit.median():.2f}, std: {legit.std():.2f}")
    print(f"  Fraud  — mean: {fraud.mean():.2f}, median: {fraud.median():.2f}, std: {fraud.std():.2f}")


# ------------------------------------------------------------------
# 4. Categorical feature breakdowns
# ------------------------------------------------------------------
def plot_categorical_fraud_rates(df: pd.DataFrame) -> None:
    """Fraud rate by age, gender, and transaction category."""
    print(f"\n{'='*60}")
    print("  CATEGORICAL FRAUD RATES")
    print(f"{'='*60}")

    cat_cols = [c for c in ["age", "gender", "category"] if c in df.columns]
    if not cat_cols:
        print("  No categorical columns found, skipping.")
        return

    n = len(cat_cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cat_cols):
        stats = df.groupby(col)[LABEL_COL].agg(["sum", "count"]).reset_index()
        stats["fraud_rate"] = stats["sum"] / stats["count"]
        stats = stats.sort_values("fraud_rate", ascending=False)

        bars = ax.bar(
            stats[col].astype(str), stats["fraud_rate"],
            color="#e74c3c", edgecolor="black", linewidth=0.3, alpha=0.85,
        )
        ax.set_title(f"Fraud Rate by {col.title()}", fontsize=13, fontweight="bold")
        ax.set_xlabel(col.title())
        ax.set_ylabel("Fraud Rate")
        ax.tick_params(axis="x", rotation=45)

        for bar, rate in zip(bars, stats["fraud_rate"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{rate:.2%}", ha="center", fontsize=8,
            )

    fig.suptitle("Fraud Rate by Categorical Features", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, "categorical_fraud_rates.png")

    for col in cat_cols:
        stats = df.groupby(col)[LABEL_COL].mean()
        print(f"\n  {col}:")
        for val, rate in stats.sort_values(ascending=False).items():
            print(f"    {val:<20s}: {rate:.4%}")


# ------------------------------------------------------------------
# 5. Correlation heatmap
# ------------------------------------------------------------------
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of feature correlations (numeric + encoded categoricals)."""
    print(f"\n{'='*60}")
    print("  CORRELATION HEATMAP")
    print(f"{'='*60}")

    df_num = df.copy()
    for col in ["age", "gender", "category"]:
        if col in df_num.columns and df_num[col].dtype == "object":
            df_num[col] = pd.factorize(df_num[col])[0]

    drop = ["customer", "merchant", "step", "zipcodeOri", "zipMerchant"]
    drop = [c for c in drop if c in df_num.columns]
    df_num = df_num.drop(columns=drop)

    numeric_cols = df_num.select_dtypes(include=[np.number]).columns.tolist()
    corr = df_num[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        square=True, linewidths=0.5, ax=ax,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, "correlation_heatmap.png")

    fraud_corr = corr[LABEL_COL].drop(LABEL_COL).sort_values(key=abs, ascending=False)
    print("  Correlation with fraud label:")
    for feat, val in fraud_corr.items():
        print(f"    {feat:<20s}: {val:+.4f}")


# ------------------------------------------------------------------
# 6. Fraud patterns — top merchants and temporal analysis
# ------------------------------------------------------------------
def plot_fraud_patterns(df: pd.DataFrame) -> None:
    """Top-fraud merchants and fraud rate over time steps."""
    print(f"\n{'='*60}")
    print("  FRAUD PATTERNS")
    print(f"{'='*60}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Top 15 merchants by fraud count
    if "merchant" in df.columns:
        merch = df[df[LABEL_COL] == 1].groupby("merchant").size().sort_values(ascending=False).head(15)
        ax = axes[0]
        ax.barh(merch.index.astype(str), merch.values, color="#e74c3c", edgecolor="black", linewidth=0.3)
        ax.set_xlabel("Fraud Transaction Count")
        ax.set_title("Top 15 Merchants by Fraud Count", fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        print(f"  Top 5 fraud merchants:")
        for m, c in merch.head(5).items():
            print(f"    {m}: {c:,} fraud txns")

    # Fraud rate per time step
    if "step" in df.columns:
        step_stats = df.groupby("step")[LABEL_COL].agg(["sum", "count"]).reset_index()
        step_stats["fraud_rate"] = step_stats["sum"] / step_stats["count"]
        ax = axes[1]
        ax.plot(step_stats["step"], step_stats["fraud_rate"], color="#e74c3c", linewidth=1.5)
        ax.fill_between(step_stats["step"], step_stats["fraud_rate"], alpha=0.2, color="#e74c3c")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Fraud Rate")
        ax.set_title("Fraud Rate Over Time", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)

    fig.suptitle("Fraud Patterns — Merchant & Temporal", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fraud_patterns.png")


# ------------------------------------------------------------------
# 7. Amount boxplot by class
# ------------------------------------------------------------------
def plot_amount_boxplot(df: pd.DataFrame) -> None:
    """Side-by-side boxplots of transaction amounts by class."""
    fig, ax = plt.subplots(figsize=(8, 5))
    df_plot = df[["amount", LABEL_COL]].copy()
    df_plot[LABEL_COL] = df_plot[LABEL_COL].map({0: "Legitimate", 1: "Fraud"})
    sns.boxplot(data=df_plot, x=LABEL_COL, y="amount", palette=["#2ecc71", "#e74c3c"], ax=ax)
    ax.set_title("Transaction Amount by Class", fontsize=15, fontweight="bold")
    ax.set_ylabel("Amount")
    ax.set_xlabel("")
    fig.tight_layout()
    _save(fig, "amount_boxplot.png")


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------
def run_eda(df: pd.DataFrame) -> None:
    """Run the complete EDA pipeline and save all plots."""
    print("\n" + "=" * 70)
    print("   EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    print(f"  Output directory: {EDA_DIR}\n")

    print_summary_statistics(df)
    plot_class_distribution(df)
    plot_amount_distributions(df)
    plot_categorical_fraud_rates(df)
    plot_correlation_heatmap(df)
    plot_fraud_patterns(df)
    plot_amount_boxplot(df)

    print(f"\n{'='*60}")
    print(f"  EDA COMPLETE — {len(os.listdir(EDA_DIR))} plots saved to {EDA_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    from preprocessing import load_data
    df = load_data()
    run_eda(df)
