"""
clustering.py
--------------
E5 - Unsupervised Analysis
K-Means and DBSCAN clustering on graph features
to detect fraud rings and anomalous groups.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from preprocessing import load_data
from graph import FraudGraph

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "EVALUATIONS", "clustering")
os.makedirs(RESULTS_DIR, exist_ok=True)

MAROON = "#8C1D40"
GOLD   = "#FFC627"
RANDOM_STATE = 42


# ---------------------------------------------------------------
# 1. Build feature matrix for clustering
# ---------------------------------------------------------------
def build_clustering_features(df):
    """Build graph-enriched feature matrix for clustering."""
    print("\n[INFO] Building clustering features...")

    fg = FraudGraph()
    df = fg.extract_degree_centrality(df)
    df = fg.extract_pagerank(df)
    df = fg.extract_merchant_fraud_rate(df)
    fg.close()

    # Select graph features for clustering
    feature_cols = [
        "customer_degree", "merchant_degree",
        "customer_pagerank", "merchant_pagerank",
        "merchant_fraud_rate", "amount"
    ]

    # Encode amount if not already scaled
    from sklearn.preprocessing import LabelEncoder
    for col in ["age", "gender", "category"]:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    feature_cols_present = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols_present].fillna(0)
    y = df["fraud"]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  Features used  : {feature_cols_present}")
    print(f"  Total samples  : {len(X_scaled):,}")

    return X_scaled, y, feature_cols_present


# ---------------------------------------------------------------
# 2. K-Means Clustering
# ---------------------------------------------------------------
def run_kmeans(X_scaled, y, n_clusters=5):
    """Run K-Means clustering and evaluate fraud concentration."""
    print(f"\n{'='*50}")
    print(f"  K-MEANS CLUSTERING (k={n_clusters})")
    print(f"{'='*50}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Silhouette score
    sil_score = silhouette_score(X_scaled, labels, sample_size=10000)
    print(f"  Silhouette Score : {sil_score:.4f}")

    # Fraud concentration per cluster
    results = pd.DataFrame({"cluster": labels, "fraud": y.values})
    cluster_stats = results.groupby("cluster")["fraud"].agg(["sum", "count", "mean"])
    cluster_stats.columns = ["fraud_count", "total", "fraud_rate"]
    cluster_stats = cluster_stats.sort_values("fraud_rate", ascending=False)

    print(f"\n  Cluster Fraud Analysis:")
    print(cluster_stats.to_string())

    high_fraud = (cluster_stats["fraud_rate"] > 0.05).sum()
    print(f"\n  Clusters with >5% fraud rate: {high_fraud}")

    return labels, cluster_stats, sil_score


# ---------------------------------------------------------------
# 3. DBSCAN Clustering
# ---------------------------------------------------------------
def run_dbscan(X_scaled, y, eps=0.5, min_samples=10):
    """Run DBSCAN to detect anomalous fraud rings."""
    print(f"\n{'='*50}")
    print(f"  DBSCAN CLUSTERING (eps={eps}, min_samples={min_samples})")
    print(f"{'='*50}")

    # Use PCA + sample for DBSCAN performance
    sample_size = 50000
    idx = np.random.choice(len(X_scaled), sample_size, replace=False)
    X_sample = X_scaled[idx]
    y_sample_vals = y.values[idx]

    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_sample)
    print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_pca)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()

    print(f"  Clusters found  : {n_clusters}")
    print(f"  Noise points    : {n_noise:,} ({n_noise/len(labels)*100:.1f}%)")

    # Fraud in noise points (anomalies)
    noise_mask  = labels == -1
    noise_fraud = y_sample_vals[noise_mask].mean() * 100
    normal_fraud = y_sample_vals[~noise_mask].mean() * 100

    print(f"  Fraud rate in noise points  : {noise_fraud:.2f}%")
    print(f"  Fraud rate in normal points : {normal_fraud:.2f}%")

    return labels, n_clusters, noise_fraud


# ---------------------------------------------------------------
# 4. Plot clustering results
# ---------------------------------------------------------------
def plot_clustering_results(X_scaled, kmeans_labels, dbscan_labels, y,
                             cluster_stats, sil_score):
    """Visualize clustering results using PCA."""

    # PCA for visualization
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled[:50000])  # sample for speed
    y_sample = y.values[:50000]
    km_sample = kmeans_labels[:50000]
    db_sample = dbscan_labels[:50000]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Clustering Analysis for Fraud Detection", fontsize=16, fontweight="bold")

    # K-Means clusters
    scatter = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1],
                                  c=km_sample, cmap="tab10",
                                  alpha=0.3, s=1)
    axes[0, 0].set_title(f"K-Means Clusters (Silhouette={sil_score:.3f})", fontsize=12)
    axes[0, 0].set_xlabel("PCA Component 1")
    axes[0, 0].set_ylabel("PCA Component 2")

    # Fraud overlay on K-Means
    fraud_mask = y_sample == 1
    axes[0, 1].scatter(X_pca[~fraud_mask, 0], X_pca[~fraud_mask, 1],
                       c="lightgray", alpha=0.2, s=1, label="Legitimate")
    axes[0, 1].scatter(X_pca[fraud_mask, 0], X_pca[fraud_mask, 1],
                       c=MAROON, alpha=0.6, s=3, label="Fraud")
    axes[0, 1].set_title("Actual Fraud Distribution", fontsize=12)
    axes[0, 1].set_xlabel("PCA Component 1")
    axes[0, 1].set_ylabel("PCA Component 2")
    axes[0, 1].legend(fontsize=10)

    # DBSCAN clusters
    axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1],
                       c=db_sample, cmap="tab10",
                       alpha=0.3, s=1)
    axes[1, 0].set_title("DBSCAN Clusters (-1 = noise/anomaly)", fontsize=12)
    axes[1, 0].set_xlabel("PCA Component 1")
    axes[1, 0].set_ylabel("PCA Component 2")

    # Cluster fraud rates bar chart
    axes[1, 1].bar(cluster_stats.index.astype(str),
                   cluster_stats["fraud_rate"] * 100,
                   color=MAROON, alpha=0.85)
    axes[1, 1].set_title("K-Means Cluster Fraud Rate (%)", fontsize=12)
    axes[1, 1].set_xlabel("Cluster")
    axes[1, 1].set_ylabel("Fraud Rate (%)")
    axes[1, 1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "clustering_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [Saved] Clustering results → {path}")


# ---------------------------------------------------------------
# 5. Cluster purity score
# ---------------------------------------------------------------
def compute_purity(labels, y):
    """Compute clustering purity score."""
    total = 0
    for cluster in set(labels):
        if cluster == -1:
            continue
        mask = labels == cluster
        if mask.sum() == 0:
            continue
        majority = max(y.values[mask].sum(),
                      mask.sum() - y.values[mask].sum())
        total += majority
    purity = total / len(y)
    print(f"\n  Clustering Purity Score: {purity:.4f}")
    return purity


# ---------------------------------------------------------------
# 6. Run full clustering pipeline
# ---------------------------------------------------------------
def run_clustering():
    print(f"\n{'='*60}")
    print("  E5 - UNSUPERVISED CLUSTERING ANALYSIS")
    print(f"{'='*60}")

    # Load data
    df = load_data()

    # Build features
    X_scaled, y, features = build_clustering_features(df)

    # K-Means
    kmeans_labels, cluster_stats, sil_score = run_kmeans(X_scaled, y, n_clusters=5)

    # DBSCAN
    dbscan_labels, n_clusters, noise_fraud = run_dbscan(X_scaled, y, eps=0.8, min_samples=50)

    # Purity
    purity = compute_purity(kmeans_labels, y)

    # Plot
    plot_clustering_results(X_scaled, kmeans_labels, dbscan_labels,
                            y, cluster_stats, sil_score)

    # Save summary
    summary = {
        "kmeans_silhouette": round(sil_score, 4),
        "kmeans_clusters": 5,
        "dbscan_clusters": n_clusters,
        "dbscan_noise_fraud_rate": round(noise_fraud, 4),
        "purity_score": round(purity, 4)
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(RESULTS_DIR, "clustering_summary.csv"), index=False)

    print(f"\n{'='*60}")
    print("  CLUSTERING SUMMARY")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    print("\n[INFO] Clustering pipeline complete!")

    return summary


if __name__ == "__main__":
    run_clustering()