# Fraud Detection with Graph Databases and Graph Neural Networks

## CSE 573 — Semantic Web Mining | Group 23 | Project 10 | Arizona State University

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

This project investigates whether graph-based structural features and Graph Neural Networks improve fraud detection performance over traditional tabular ML models on the BankSim banking transaction dataset.

**Core Hypothesis:** Graph-derived structure reveals fraud patterns that flat feature vectors miss.

**Result:** ✅ Hypothesis confirmed — every model improved when graph features were added.

---

## Repository Structure

```
fraud-detection/
├── preprocessing.py        ← Data loading, SMOTE-ENN, time-based split
├── eda.py                  ← 7 EDA visualizations
├── models.py               ← E1: Baseline ML models with hyperparameter tuning
├── graph.py                ← E2: Neo4j graph construction + feature extraction
├── hybrid_models.py        ← E2: Graph-enhanced ML models
├── graphsage.py            ← E4: Standalone GraphSAGE classifier
├── embedding_models.py     ← E3: GraphSAGE embeddings → RF + MLP
├── clustering.py           ← E5: K-Means + DBSCAN fraud ring detection
├── main.py                 ← Runs full pipeline end to end
├── dashboard/              ← React dashboard for results visualization
│   ├── src/App.js          ← Main dashboard component
│   └── public/images/      ← All result plots served to dashboard
├── DATA/                   ← Dataset (auto-downloaded via kagglehub)
├── EVALUATIONS/            ← All results, plots, and CSVs
│   ├── eda/                ← EDA plots
│   └── clustering/         ← Clustering plots
├── results/                ← Generated outputs from pipeline
└── README.md
```

---

## Dataset

**BankSim** — Synthetic banking transaction data from Kaggle

| Property | Value |
|---|---|
| Total Transactions | 594,643 |
| Fraudulent | 7,200 (1.21%) |
| Legitimate | 587,443 (98.79%) |
| Imbalance Ratio | 1:81 |
| Simulation Period | ~6 months |
| Source | kaggle.com/datasets/ealaxi/banksim1 |

---

## Experiments & Results

| Experiment | Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| E1 — Baseline | Logistic Regression | 0.9491 | 0.8664 | 0.9059 | 0.9714 |
| E1 — Baseline | Random Forest (Tuned) | 0.9867 | 1.0000 | 0.9933 | 0.9996 |
| E1 — Baseline | XGBoost (Tuned) | 0.9524 | 1.0000 | 0.9756 | 0.9966 |
| E2 — Graph-Enhanced | LR + Graph Features | 0.9682 | 0.9740 | 0.9711 | 0.9959 |
| E2 — Graph-Enhanced | RF + Graph Features | 0.9976 | 1.0000 | 0.9988 | **1.0000** |
| E2 — Graph-Enhanced | XGBoost + Graph Features | 0.9792 | 1.0000 | 0.9895 | 0.9993 |
| E3 — Embeddings | GraphSAGE + RF | 0.9963 | 1.0000 | 0.9981 | 0.9997 |
| E3 — Embeddings | GraphSAGE + MLP | 0.9933 | 1.0000 | 0.9966 | 0.9987 |
| E4 — Standalone GNN | GraphSAGE | 0.4344 | 0.9298 | 0.5922 | 0.7143 |
| E5 — Clustering | K-Means Purity | — | — | — | 0.9916 |

---

## Key Findings

- **Graph features improve every model** — Logistic Regression recall jumped from 0.87 → 0.97
- **11 out of 49 merchants** have >50% fraud rate — invisible without graph analysis
- **K-Means Cluster 4** — 337 transactions, 99.7% fraud rate — pure fraud ring detected without labels
- **DBSCAN noise points** — 65.2% fraud rate vs 0.23% in normal clusters
- **GraphSAGE embeddings** as features for RF achieves near-perfect PR-AUC of 0.9997
- **Amount** has 0.49 correlation with fraud — strongest single tabular predictor
- **es_leisure** and **es_travel** categories have 80%+ fraud rates

---

## How to Run

### Prerequisites
- Python 3.10+
- Node.js 16+ (for dashboard)
- Neo4j (for graph features)

### 1. Install Python dependencies
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
python main.py
```

Or run individual steps:
```bash
python preprocessing.py    # Step 1 — Data prep
python eda.py              # Step 2 — EDA
python models.py           # Step 3 — E1 Baseline
python graph.py            # Step 4 — Graph construction
python hybrid_models.py    # Step 5 — E2 Hybrid
python graphsage.py        # Step 6 — E4 GraphSAGE
python embedding_models.py # Step 7 — E3 Embeddings
python clustering.py       # Step 8 — E5 Clustering
```

### 4. Run the React Dashboard
```bash
cd dashboard
npm install
npm start
```

Then open `http://localhost:3000` in your browser.

The dashboard shows:
- Overview with key stats and hypothesis result
- Individual tabs for each experiment (E1–E5)
- EDA insights and fraud pattern analysis
- Results gallery with all generated plots

---

## Graph Construction Details

- **Type:** Bipartite graph — customers on one side, merchants on the other
- **Nodes:** 4,061 customers + 49 merchants = 4,110 total
- **Edges:** Each transaction = one directed edge from Customer → Merchant
- **Edge weight:** Transaction amount
- **Multiple transactions:** Kept as separate time-ordered edges
- **Time-based split:** Uses `step` column — earlier steps for training, later steps for testing (prevents data leakage)

---

## Graph Features Extracted

| Feature | Description |
|---|---|
| Degree Centrality | Number of transactions per customer/merchant |
| PageRank | Importance/centrality of each node in the network |
| Betweenness Centrality | How often a node sits between other nodes |
| Community Detection | Which cluster/group each node belongs to |
| Merchant Fraud Rate | % of each merchant's transactions that are fraudulent |

---

## Evaluation Metrics

| Metric | Why Used |
|---|---|
| **PR-AUC** | Primary metric — best for rare event detection under severe imbalance |
| **Recall** | Critical — missing fraud is costly |
| **Precision** | Ensures flagged fraud is actually fraud |
| **F1-Score** | Balance between Precision and Recall |
| **Purity** | Clustering quality metric |

Accuracy is deliberately excluded — with 1.2% fraud, a model predicting everything as legitimate gets 98.8% accuracy while catching zero fraud.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| scikit-learn | ML models + preprocessing |
| XGBoost | Gradient boosting baseline |
| imbalanced-learn | SMOTE-ENN class balancing |
| Neo4j | Graph database |
| networkx | Betweenness centrality computation |
| PyTorch | Deep learning |
| PyTorch Geometric | GraphSAGE implementation |
| pandas / numpy | Data manipulation |
| matplotlib / seaborn | Visualizations |
| React | Interactive results dashboard |
| Git / GitHub | Version control |

---

## References

1. Kaggle — BankSim1 Dataset: https://www.kaggle.com/datasets/ealaxi/banksim1
2. Neo4j Developer Guides: https://neo4j.com/developer/get-started/
3. Alloy — 2024 Fraud Stats: https://www.alloy.com/blog/2024-fraud-stats-for-banks-fintechs-and-credit-unions
4. Mathematics (MDPI), vol. 11, no. 13, Article 2862: https://www.mdpi.com/2227-7390/11/13/2862
5. Wang et al., "A Semi-Supervised Graph Attentive Network for Financial Fraud Detection," IEEE ICDM 2019. DOI: 10.1109/ICDM.2019.00070
