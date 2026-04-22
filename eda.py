"""
eda.py
-------
Exploratory Data Analysis for BankSim fraud detection dataset.
Generates visualizations showing fraud patterns by category,
amount, time, merchant, and customer demographics.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_data

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "eda")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ASU colors
MAROON = "#8C1D40"
GOLD   = "#FFC627"
GRAY   = "#64748B"


# ---------------------------------------------------------------
# 1. Class Distribution
# ---------------------------------------------------------------
def plot_class_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Distribution — Fraud vs Legitimate", fontsize=15, fontweight="bold")

    # Count plot
    counts = df["fraud"].value_counts()
    axes[0].bar(["Legitimate", "Fraud"], counts.values,
                color=[GRAY, MAROON], alpha=0.85)
    axes[0].set_title("Transaction Counts", fontsize=13)
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 1000, f"{v:,}", ha="center", fontweight="bold")

    # Pie chart
    axes[1].pie(counts.values, labels=["Legitimate (98.8%)", "Fraud (1.2%)"],
                colors=[GRAY, MAROON], autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Class Proportion", fontsize=13)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "01_class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] Class distribution → {path}")


# ---------------------------------------------------------------
# 2. Fraud by Transaction Category
# ---------------------------------------------------------------
def plot_fraud_by_category(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Fraud Analysis by Transaction Category", fontsize=15, fontweight="bold")

    # Fraud count by category
    cat_fraud = df.groupby("category")["fraud"].agg(["sum", "count"]).reset_index()
    cat_fraud.columns = ["category", "fraud_count", "total"]
    cat_fraud["fraud_rate"] = cat_fraud["fraud_count"] / cat_fraud["total"] * 100
    cat_fraud = cat_fraud.sort_values("fraud_rate", ascending=True)

    axes[0].barh(cat_fraud["category"], cat_fraud["fraud_count"],
                 color=MAROON, alpha=0.85)
    axes[0].set_title("Fraud Count by Category", fontsize=13)
    axes[0].set_xlabel("Number of Fraud Transactions")

    axes[1].barh(cat_fraud["category"], cat_fraud["fraud_rate"],
                 color=GOLD, alpha=0.85)
    axes[1].set_title("Fraud Rate by Category (%)", fontsize=13)
    axes[1].set_xlabel("Fraud Rate (%)")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "02_fraud_by_category.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] Fraud by category → {path}")


# ---------------------------------------------------------------
# 3. Transaction Amount Distribution
# ---------------------------------------------------------------
def plot_amount_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Transaction Amount Distribution", fontsize=15, fontweight="bold")

    fraud_amounts = df[df["fraud"] == 1]["amount"]
    legit_amounts = df[df["fraud"] == 0]["amount"]

    axes[0].hist(legit_amounts, bins=50, color=GRAY, alpha=0.7, label="Legitimate")
    axes[0].hist(fraud_amounts, bins=50, color=MAROON, alpha=0.7, label="Fraud")
    axes[0].set_title("Amount Distribution (Full Range)", fontsize=13)
    axes[0].set_xlabel("Transaction Amount")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # Zoomed in
    axes[1].hist(legit_amounts[legit_amounts < 2000], bins=50, color=GRAY, alpha=0.7, label="Legitimate")
    axes[1].hist(fraud_amounts[fraud_amounts < 2000], bins=50, color=MAROON, alpha=0.7, label="Fraud")
    axes[1].set_title("Amount Distribution (< $2000)", fontsize=13)
    axes[1].set_xlabel("Transaction Amount")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "03_amount_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] Amount distribution → {path}")


# ---------------------------------------------------------------
# 4. Fraud Over Time (step column)
# ---------------------------------------------------------------
def plot_fraud_over_time(df):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Fraud Patterns Over Time", fontsize=15, fontweight="bold")

    time_data = df.groupby("step")["fraud"].agg(["sum", "count"]).reset_index()
    time_data.columns = ["step", "fraud_count", "total"]
    time_data["fraud_rate"] = time_data["fraud_count"] / time_data["total"] * 100

    axes[0].plot(time_data["step"], time_data["fraud_count"],
                 color=MAROON, linewidth=1.5)
    axes[0].fill_between(time_data["step"], time_data["fraud_count"],
                         alpha=0.3, color=MAROON)
    axes[0].set_title("Fraud Count per Time Step", fontsize=13)
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Fraud Count")
    axes[0].grid(alpha=0.3)

    axes[1].plot(time_data["step"], time_data["fraud_rate"],
                 color=GOLD, linewidth=1.5)
    axes[1].fill_between(time_data["step"], time_data["fraud_rate"],
                         alpha=0.3, color=GOLD)
    axes[1].set_title("Fraud Rate (%) per Time Step", fontsize=13)
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Fraud Rate (%)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "04_fraud_over_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] Fraud over time → {path}")


# ---------------------------------------------------------------
# 5. Fraud by Customer Age and Gender
# ---------------------------------------------------------------
def plot_fraud_by_demographics(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Fraud by Customer Demographics", fontsize=15, fontweight="bold")

    # By age
    age_fraud = df.groupby("age")["fraud"].agg(["sum", "count"]).reset_index()
    age_fraud.columns = ["age", "fraud_count", "total"]
    age_fraud["fraud_rate"] = age_fraud["fraud_count"] / age_fraud["total"] * 100
    age_fraud = age_fraud.sort_values("age")

    axes[0].bar(age_fraud["age"].astype(str), age_fraud["fraud_rate"],
                color=MAROON, alpha=0.85)
    axes[0].set_title("Fraud Rate by Age Group", fontsize=13)
    axes[0].set_xlabel("Age Group")
    axes[0].set_ylabel("Fraud Rate (%)")
    axes[0].grid(axis="y", alpha=0.3)

    # By gender
    gender_fraud = df.groupby("gender")["fraud"].agg(["sum", "count"]).reset_index()
    gender_fraud.columns = ["gender", "fraud_count", "total"]
    gender_fraud["fraud_rate"] = gender_fraud["fraud_count"] / gender_fraud["total"] * 100

    axes[1].bar(gender_fraud["gender"], gender_fraud["fraud_rate"],
                color=GOLD, alpha=0.85)
    axes[1].set_title("Fraud Rate by Gender", fontsize=13)
    axes[1].set_xlabel("Gender")
    axes[1].set_ylabel("Fraud Rate (%)")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "05_fraud_by_demographics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] Fraud by demographics → {path}")


# ---------------------------------------------------------------
# 6. Top Merchants by Fraud Rate
# ---------------------------------------------------------------
def plot_merchant_fraud(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Merchant Fraud Analysis", fontsize=15, fontweight="bold")

    merchant_stats = df.groupby("merchant")["fraud"].agg(["sum", "count"]).reset_index()
    merchant_stats.columns = ["merchant", "fraud_count", "total"]
    merchant_stats["fraud_rate"] = merchant_stats["fraud_count"] / merchant_stats["total"] * 100

    # Top 15 by fraud count
    top_count = merchant_stats.nlargest(15, "fraud_count")
    axes[0].barh(range(len(top_count)), top_count["fraud_count"],
                 color=MAROON, alpha=0.85)
    axes[0].set_yticks(range(len(top_count)))
    axes[0].set_yticklabels([f"M{i+1}" for i in range(len(top_count))], fontsize=9)
    axes[0].set_title("Top 15 Merchants by Fraud Count", fontsize=13)
    axes[0].set_xlabel("Fraud Count")

    # Top 15 by fraud rate
    top_rate = merchant_stats.nlargest(15, "fraud_rate")
    axes[1].barh(range(len(top_rate)), top_rate["fraud_rate"],
                 color=GOLD, alpha=0.85)
    axes[1].set_yticks(range(len(top_rate)))
    axes[1].set_yticklabels([f"M{i+1}" for i in range(len(top_rate))], fontsize=9)
    axes[1].set_title("Top 15 Merchants by Fraud Rate (%)", fontsize=13)
    axes[1].set_xlabel("Fraud Rate (%)")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "06_merchant_fraud.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] Merchant fraud → {path}")


# ---------------------------------------------------------------
# 7. Correlation Heatmap
# ---------------------------------------------------------------
def plot_correlation_heatmap(df):
    from sklearn.preprocessing import LabelEncoder

    df_enc = df.copy()
    for col in ["age", "gender", "category", "customer", "merchant"]:
        if col in df_enc.columns:
            df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

    corr = df_enc.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "07_correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] Correlation heatmap → {path}")


# ---------------------------------------------------------------
# 8. Run all EDA
# ---------------------------------------------------------------
def run_eda():
    print(f"\n{'='*60}")
    print("  EXPLORATORY DATA ANALYSIS")
    print(f"{'='*60}")

    df = load_data()

    print("\n[INFO] Generating EDA visualizations...")
    plot_class_distribution(df)
    plot_fraud_by_category(df)
    plot_amount_distribution(df)
    plot_fraud_over_time(df)
    plot_fraud_by_demographics(df)
    plot_merchant_fraud(df)
    plot_correlation_heatmap(df)

    print(f"\n[INFO] All EDA plots saved to {RESULTS_DIR}")
    print("[INFO] EDA complete!")


if __name__ == "__main__":
    run_eda()