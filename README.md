# Fraud Detection with Graph Databases and Graph Neural Networks

## CSE 573 — Semantic Web Mining | Group 23 | Project 10

### Team Members

| Name | Email | ASU ID |
|------|-------|--------|
| Luan Nguyen | ltnguy58@asu.edu | 1225177265 |
| Dhanush Mugajji Shambulingappa | dmugajji@asu.edu | 1237526292 |
| Chandra Shekhar Pavuluri | cpavulur@asu.edu | 1236078196 |
| Adrian Zhang | awzhang1@asu.edu | 1224664415 |
| Shashikant Nanda | snanda5@asu.edu | 1235508926 |
| Chitwandeep Kaur Palne | cpalne@asu.edu | 1231519031 |

---

## Project Overview

This project investigates whether graph-based structural features and Graph Neural Networks improve fraud detection performance over traditional tabular ML models.

**Core Hypothesis:** Graph-derived structure reveals fraud patterns that flat feature vectors miss.

**Dataset:** BankSim — 594,643 synthetic banking transactions with 1.2% fraud rate.

---

## Repository Structure

fraud-detection/
├── CODE/               ← All Python source files
├── DATA/               ← Dataset (auto-downloaded via kagglehub)
├── EVALUATIONS/        ← All results, plots, and CSVs
│   ├── eda/            ← Exploratory Data Analysis plots
│   └── clustering/     ← Clustering analysis plots
├── results/            ← Generated outputs
├── preprocessing.py    ← Data loading, SMOTE-ENN, time-based split
├── models.py           ← E1: Baseline ML models
├── graph.py            ← E2: Neo4j graph construction + feature extraction
├── hybrid_models.py    ← E2: Graph-enhanced ML models
├── graphsage.py        ← E4: Standalone GraphSAGE classifier
├── embedding_models.py ← E3: GraphSAGE embeddings → RF + MLP
├── clustering.py       ← E5: K-Means + DBSCAN fraud ring detection
└── eda.py              ← Exploratory Data Analysis

---

## Experiments

| Experiment | Description | Key Result |
|---|---|---|
| E1 — Tabular Baseline | LR, RF, XGBoost on raw features | RF PR-AUC: 0.999 |
| E2 — Graph-Enhanced ML | Neo4j graph metrics + tabular features | RF PR-AUC: 1.000 |
| E3 — Embedding Hybrid | GraphSAGE embeddings → RF + MLP | RF PR-AUC: 0.9997 |
| E4 — Standalone GraphSAGE | End-to-end GNN classifier | PR-AUC: 0.714 |
| E5 — Unsupervised | K-Means + DBSCAN clustering | Purity: 0.9916 |

---

## Key Findings

- **Graph features improve every model** — Logistic Regression recall jumped from 0.87 → 0.97 with graph features
- **11 out of 49 merchants** have >50% fraud rate — invisible without graph analysis
- **DBSCAN noise points** have 65% fraud rate vs 0.23% in normal clusters — anomaly detection works
- **K-Means Cluster 4** — 337 transactions, 99.7% fraud rate — pure fraud ring detected

---

## How to Run

### Prerequisites
- Python 3.10+
- Neo4j

### 1. Install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch torch-geometric networkx seaborn
```

### 2. Start Neo4j
```bash
/opt/homebrew/opt/neo4j/bin/neo4j start
```

### 3. Run the full pipeline
```bash
# Step 1 — Preprocessing + SMOTE-ENN + Time-based split
python preprocessing.py

# Step 2 — EDA visualizations
python eda.py

# Step 3 — Baseline ML models (E1)
python models.py

# Step 4 — Graph construction + feature extraction (E2)
python graph.py

# Step 5 — Hybrid models (E2)
python hybrid_models.py

# Step 6 — GraphSAGE standalone (E4)
python graphsage.py

# Step 7 — Embedding-based models (E3)
python embedding_models.py

# Step 8 — Clustering analysis (E5)
python clustering.py
```

---

## Graph Construction Details

- **Nodes:** Customers and Merchants
- **Edges:** Each transaction = one directed edge from Customer → Merchant
- **Edge weight:** Transaction amount
- **Multiple transactions:** Kept as separate time-ordered edges
- **Time-based split:** Uses `step` column to prevent data leakage — earlier steps for training, later steps for testing

---

## Evaluation Metrics

- **Precision** — Of flagged fraud, how many are actually fraud?
- **Recall** — Of all real fraud cases, how many did we catch?
- **F1-Score** — Balance between Precision and Recall
- **PR-AUC** — Primary metric for imbalanced fraud detection

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| scikit-learn | ML models + preprocessing |
| XGBoost | Gradient boosting baseline |
| Neo4j | Graph database |
| PyTorch Geometric | GraphSAGE implementation |
| NetworkX | Betweenness centrality |
| SMOTE-ENN | Class imbalance handling |
| pandas / numpy | Data manipulation |
| matplotlib / seaborn | Visualizations |