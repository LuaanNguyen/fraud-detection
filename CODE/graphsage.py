"""
graphsage.py
-------------
GraphSAGE model for fraud detection.
Builds a transaction graph, trains GraphSAGE,
and evaluates as a standalone node classifier.
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
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, classification_report,
    precision_recall_curve
)

from preprocessing import load_data

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "EVALUATIONS")
os.makedirs(RESULTS_DIR, exist_ok=True)
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)


# ---------------------------------------------------------------
# 1. Build PyTorch Geometric Graph
# ---------------------------------------------------------------
def build_pyg_graph(df, sample_size=30000):
    """
    Convert BankSim transactions into a PyTorch Geometric graph.
    Nodes = Customers + Merchants
    Edges = Transactions
    """
    print(f"\n[INFO] Building PyG graph with {sample_size:,} transactions...")

    df = df.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)

    # Encode categoricals
    le = LabelEncoder()
    for col in ["age", "gender", "category"]:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    scaler = StandardScaler()
    df["amount"] = scaler.fit_transform(df[["amount"]])

    # Create unified node index
    customers = df["customer"].unique()
    merchants = df["merchant"].unique()

    customer_idx = {c: i for i, c in enumerate(customers)}
    merchant_idx = {m: i + len(customers) for i, m in enumerate(merchants)}

    num_nodes = len(customers) + len(merchants)
    print(f"  Nodes: {len(customers):,} customers + {len(merchants):,} merchants = {num_nodes:,} total")

    # Build edge index
    src = [customer_idx[c] for c in df["customer"]]
    dst = [merchant_idx[m] for m in df["merchant"]]

    # Bidirectional edges
    edge_index = torch.tensor(
        [src + dst, dst + src], dtype=torch.long
    )

    # Node features
    # Customer features: mean age, gender, amount per customer
    customer_features = df.groupby("customer")[["age", "gender", "amount"]].mean()
    merchant_features = df.groupby("merchant")[["category", "amount"]].mean()

    # Build feature matrix
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

    # Node labels — fraud flag per customer
    # A customer is "fraud" if any of their transactions are fraudulent
    customer_fraud = df.groupby("customer")["fraud"].max()
    merchant_fraud = df.groupby("merchant")["fraud"].max()

    labels = np.zeros(num_nodes, dtype=int)
    for cust, idx in customer_idx.items():
        if cust in customer_fraud.index:
            labels[idx] = customer_fraud[cust]
    for merch, idx in merchant_idx.items():
        if merch in merchant_fraud.index:
            labels[idx] = merchant_fraud[merch]

    y = torch.tensor(labels, dtype=torch.long)

    # Train/test masks (80/20 split)
    indices = torch.randperm(num_nodes)
    train_size = int(0.8 * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:train_size]] = True
    test_mask[indices[train_size:]]  = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        test_mask=test_mask
    )

    print(f"  Edges          : {edge_index.shape[1]:,}")
    print(f"  Fraud nodes    : {y.sum().item():,} / {num_nodes:,}")
    print(f"  Train nodes    : {train_mask.sum().item():,}")
    print(f"  Test nodes     : {test_mask.sum().item():,}")

    return data, customer_idx, merchant_idx


# ---------------------------------------------------------------
# 2. GraphSAGE Model
# ---------------------------------------------------------------
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3 - output
        x = self.conv3(x, edge_index)
        return x


# ---------------------------------------------------------------
# 3. Train GraphSAGE
# ---------------------------------------------------------------
def train_graphsage(data, epochs=100, lr=0.01, hidden=64):
    """Train the GraphSAGE model."""
    print(f"\n[INFO] Training GraphSAGE...")
    print(f"  Epochs  : {epochs}")
    print(f"  LR      : {lr}")
    print(f"  Hidden  : {hidden}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device  : {device}")

    model = GraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=hidden,
        out_channels=2
    ).to(device)

    data = data.to(device)

    # Class weights for imbalance
    fraud_count = data.y[data.train_mask].sum().item()
    legit_count = data.train_mask.sum().item() - fraud_count
    weight = torch.tensor(
        [1.0, legit_count / max(fraud_count, 1)],
        dtype=torch.float
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)

    train_losses = []
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        out  = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:>3} | Loss: {loss.item():.4f}")

    # Load best model
    model.load_state_dict(best_state)

    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, color="#8C1D40", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GraphSAGE Training Loss", fontweight="bold")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(RESULTS_DIR, "graphsage_loss.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"  [Saved] Training loss → {loss_path}")

    return model, data


# ---------------------------------------------------------------
# 4. Evaluate GraphSAGE
# ---------------------------------------------------------------
def evaluate_graphsage(model, data):
    """Evaluate the trained GraphSAGE model."""
    print(f"\n{'='*60}")
    print("  GRAPHSAGE EVALUATION")
    print(f"{'='*60}")

    model.eval()
    with torch.no_grad():
        out    = model(data.x, data.edge_index)
        probs  = F.softmax(out, dim=1)
        y_scores = probs[data.test_mask][:, 1].cpu().numpy()
        y_pred   = out[data.test_mask].argmax(dim=1).cpu().numpy()
        y_true   = data.y[data.test_mask].cpu().numpy()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    pr_auc    = average_precision_score(y_true, y_scores)

    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  PR-AUC    : {pr_auc:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Legit','Fraud'])}")

    # PR Curve
    p_vals, r_vals, _ = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(8, 5))
    plt.plot(r_vals, p_vals, color="#8C1D40", linewidth=2,
             label=f"GraphSAGE (PR-AUC={pr_auc:.3f})")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("GraphSAGE Precision-Recall Curve", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    pr_path = os.path.join(RESULTS_DIR, "graphsage_pr_curve.png")
    plt.savefig(pr_path, dpi=150)
    plt.close()
    print(f"  [Saved] PR curve → {pr_path}")

    return {
        "Model"    : "GraphSAGE",
        "Precision": round(precision, 4),
        "Recall"   : round(recall, 4),
        "F1-Score" : round(f1, 4),
        "PR-AUC"   : round(pr_auc, 4),
    }


# ---------------------------------------------------------------
# 5. Final comparison — all models
# ---------------------------------------------------------------
def plot_final_comparison(graphsage_result):
    """Compare all models — baseline, hybrid, GraphSAGE."""
    baseline_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
    hybrid_path   = os.path.join(RESULTS_DIR, "hybrid_results.csv")

    if not os.path.exists(baseline_path) or not os.path.exists(hybrid_path):
        print("[WARNING] Baseline or hybrid results not found, skipping comparison")
        return

    baseline_df = pd.read_csv(baseline_path)
    hybrid_df   = pd.read_csv(hybrid_path)

    # Remove SVM
    baseline_df = baseline_df[baseline_df["Model"] != "SVM"].reset_index(drop=True)

    # Rename for clarity
    baseline_df["Model"] = baseline_df["Model"] + " (Baseline)"
    hybrid_df["Model"]   = hybrid_df["Model"] + " (Graph)"

    # Add GraphSAGE
    gs_df = pd.DataFrame([graphsage_result])

    all_results = pd.concat([baseline_df, hybrid_df, gs_df], ignore_index=True)

    # Save full comparison
    all_results.to_csv(os.path.join(RESULTS_DIR, "full_comparison.csv"), index=False)

    # Plot
    metrics = ["Precision", "Recall", "F1-Score", "PR-AUC"]
    colors  = ["#8C1D40"] * 3 + ["#FFC627"] * 3 + ["#1A6B3C"]

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(all_results))
    for i, metric in enumerate(metrics):
        ax.plot(x, all_results[metric], marker="o", linewidth=2, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(all_results["Model"], rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0.7, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Full Model Comparison: Baseline → Graph-Enhanced → GraphSAGE",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "full_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  [Saved] Full comparison → {path}")
    print(all_results.to_string(index=False))


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------
def run_graphsage():
    df   = load_data()
    data, _, _ = build_pyg_graph(df, sample_size=30000)
    model, data = train_graphsage(data, epochs=100, lr=0.01, hidden=64)
    result = evaluate_graphsage(model, data)
    plot_final_comparison(result)
    print("\n[INFO] GraphSAGE pipeline complete!")
    return result


if __name__ == "__main__":
    run_graphsage()