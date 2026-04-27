"""
embedding_models.py
--------------------
E3 - Embedding-based Hybrid Models
Uses GraphSAGE embeddings as input features
for XGBoost and MLP classifiers.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, classification_report,
    precision_recall_curve
)
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from preprocessing import load_data

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "EVALUATIONS")
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)

MAROON = "#8C1D40"
GOLD   = "#FFC627"


# ---------------------------------------------------------------
# 1. Build PyG graph (reused from graphsage.py)
# ---------------------------------------------------------------
def build_pyg_graph(df, sample_size=30000):
    """Build PyTorch Geometric graph from transaction data."""
    print(f"\n[INFO] Building graph for embeddings ({sample_size:,} transactions)...")

    df = df.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)

    le = LabelEncoder()
    for col in ["age", "gender", "category"]:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    scaler = StandardScaler()
    df["amount"] = scaler.fit_transform(df[["amount"]])

    customers = df["customer"].unique()
    merchants = df["merchant"].unique()
    customer_idx = {c: i for i, c in enumerate(customers)}
    merchant_idx = {m: i + len(customers) for i, m in enumerate(merchants)}
    num_nodes = len(customers) + len(merchants)

    src = [customer_idx[c] for c in df["customer"]]
    dst = [merchant_idx[m] for m in df["merchant"]]
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

    # Node features
    customer_features = df.groupby("customer")[["age", "gender", "amount"]].mean()
    merchant_features = df.groupby("merchant")[["category", "amount"]].mean()

    node_features = np.zeros((num_nodes, 3))
    for cust, idx in customer_idx.items():
        if cust in customer_features.index:
            row = customer_features.loc[cust]
            node_features[idx] = [row["age"], row["gender"], row["amount"]]
    for merch, idx in merchant_idx.items():
        if merch in merchant_features.index:
            row = merchant_features.loc[merch]
            node_features[idx] = [row["category"], 0, row["amount"]]

    x = torch.tensor(node_features, dtype=torch.float)

    # Labels
    customer_fraud = df.groupby("customer")["fraud"].max()
    merchant_fraud  = df.groupby("merchant")["fraud"].max()
    labels = np.zeros(num_nodes, dtype=int)
    for cust, idx in customer_idx.items():
        if cust in customer_fraud.index:
            labels[idx] = customer_fraud[cust]
    for merch, idx in merchant_idx.items():
        if merch in merchant_fraud.index:
            labels[idx] = merchant_fraud[merch]

    y = torch.tensor(labels, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    print(f"  Nodes: {num_nodes:,} | Edges: {edge_index.shape[1]:,}")

    return data, customer_idx, merchant_idx, df


# ---------------------------------------------------------------
# 2. GraphSAGE Encoder (embedding generator)
# ---------------------------------------------------------------
class GraphSAGEEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, embedding_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, embedding_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# ---------------------------------------------------------------
# 3. Train encoder and extract embeddings
# ---------------------------------------------------------------
def train_and_extract_embeddings(data, epochs=80, hidden=64, embedding_dim=32):
    """Train GraphSAGE encoder and extract node embeddings."""
    print(f"\n[INFO] Training GraphSAGE encoder for embeddings...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = GraphSAGEEncoder(
        in_channels=data.x.shape[1],
        hidden_channels=hidden,
        embedding_dim=embedding_dim
    ).to(device)

    data = data.to(device)

    # Class weights
    fraud_count = data.y.sum().item()
    legit_count = len(data.y) - fraud_count
    weight = torch.tensor(
        [1.0, legit_count / max(fraud_count, 1)],
        dtype=torch.float
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)

    # Simple train mask
    train_mask = torch.zeros(len(data.y), dtype=torch.bool)
    train_mask[:int(0.8 * len(data.y))] = True
    train_mask = train_mask.to(device)

    # Add output layer for training
    out_layer = torch.nn.Linear(embedding_dim, 2).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(out_layer.parameters()),
        lr=0.01, weight_decay=5e-4
    )

    for epoch in range(1, epochs + 1):
        model.train()
        out_layer.train()
        optimizer.zero_grad()
        embeddings = model(data.x, data.edge_index)
        out  = out_layer(embeddings)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:>3} | Loss: {loss.item():.4f}")

    # Extract embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index).cpu().numpy()

    print(f"  Embeddings shape: {embeddings.shape}")
    return embeddings, data.y.cpu().numpy()


# ---------------------------------------------------------------
# 4. Map embeddings back to transactions
# ---------------------------------------------------------------
def map_embeddings_to_transactions(embeddings, customer_idx, merchant_idx, df_sample):
    """Map node embeddings back to transaction-level features."""
    print("\n[INFO] Mapping embeddings to transactions...")

    emb_features = []
    for _, row in df_sample.iterrows():
        c_idx = customer_idx.get(row["customer"], -1)
        m_idx = merchant_idx.get(row["merchant"], -1)

        if c_idx >= 0 and m_idx >= 0:
            # Concatenate customer and merchant embeddings
            c_emb = embeddings[c_idx]
            m_emb = embeddings[m_idx]
            emb_features.append(np.concatenate([c_emb, m_emb]))
        else:
            emb_features.append(np.zeros(embeddings.shape[1] * 2))

    X_emb = np.array(emb_features)
    y_emb = df_sample["fraud"].values

    print(f"  Transaction embedding matrix: {X_emb.shape}")
    return X_emb, y_emb


# ---------------------------------------------------------------
# 5. Evaluate model
# ---------------------------------------------------------------
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train and evaluate a classifier on embeddings."""
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")

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
# 6. Plot PR curves
# ---------------------------------------------------------------
def plot_pr_curves(results, y_test):
    """Plot PR curves for embedding-based models."""
    plt.figure(figsize=(8, 5))
    colors = [MAROON, GOLD]

    for i, res in enumerate(results):
        p, r, _ = precision_recall_curve(y_test, res["y_scores"])
        plt.plot(r, p, label=f"{res['Model']} (PR-AUC={res['PR-AUC']:.3f})",
                 color=colors[i], linewidth=2)

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("E3 — Embedding-Based Models PR Curves", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "e3_embedding_pr_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  [Saved] PR curves → {path}")


# ---------------------------------------------------------------
# 7. Full E3 pipeline
# ---------------------------------------------------------------
def run_embedding_models():
    print(f"\n{'='*60}")
    print("  E3 - GRAPHSAGE EMBEDDING-BASED HYBRID MODELS")
    print(f"{'='*60}")

    # Load data
    df = load_data()

    # Build graph
    data, customer_idx, merchant_idx, df_sample = build_pyg_graph(df, sample_size=30000)

    # Train encoder and get embeddings
    embeddings, node_labels = train_and_extract_embeddings(
        data, epochs=80, hidden=64, embedding_dim=32
    )

    # Map to transactions
    X_emb, y_emb = map_embeddings_to_transactions(
        embeddings, customer_idx, merchant_idx, df_sample
    )

    # Balance with oversampling
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_bal, y_bal = ros.fit_resample(X_emb, y_emb)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2,
        stratify=y_bal, random_state=RANDOM_STATE
    )

    print(f"\n  Training set : {X_train.shape[0]:,} samples")
    print(f"  Test set     : {X_test.shape[0]:,} samples")
    print(f"  Embedding dim: {X_train.shape[1]}")

    # Define models
    models = [
        ("GraphSAGE Embeddings + Random Forest", RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        )),
        ("GraphSAGE Embeddings + MLP", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=200, random_state=RANDOM_STATE
        )),
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
    print("  E3 EMBEDDING MODELS SUMMARY")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))

    # Save
    csv_path = os.path.join(RESULTS_DIR, "e3_embedding_results.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  [Saved] Results → {csv_path}")

    # Plot
    plot_pr_curves(results, y_test)

    print("\n[INFO] E3 embedding pipeline complete!")
    return summary_df


if __name__ == "__main__":
    run_embedding_models()