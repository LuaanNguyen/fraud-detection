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

**Result:** Hypothesis confirmed — every model improved when graph features were added.

---

## Repository Structure

```
fraud-detection/
├── main.py                 ← Runs full 8-step pipeline end to end
├── preprocessing.py        ← Data loading, cleaning, SMOTE-ENN, time-based split
├── eda.py                  ← 7 EDA visualizations (class dist, fraud patterns, etc.)
├── models.py               ← E1: Baseline ML (LR, SVM, RF, XGBoost) + tuning
├── graph.py                ← E2: Neo4j graph construction + feature extraction
├── hybrid_models.py        ← E2: Graph-enhanced ML models
├── graphsage.py            ← E4: Standalone GraphSAGE node classifier
├── embedding_models.py     ← E3: GraphSAGE embeddings → RF + MLP
├── clustering.py           ← E5: K-Means + DBSCAN fraud ring detection
├── requirements.txt        ← Python dependencies
├── dashboard/              ← React dashboard for results visualization
│   ├── src/App.js          ← Main dashboard component
│   └── public/images/      ← All result plots served to dashboard
├── data/                   ← Dataset (auto-downloaded via kagglehub)
├── results/                ← Generated outputs from pipeline runs
├── EVALUATIONS/            ← All results, plots, and CSVs
│   ├── eda/                ← EDA plots
│   └── clustering/         ← Clustering plots
├── ARCHITECTURE.md         ← Data flow, module details, design decisions
└── README.md
```

---

## Dataset

**BankSim** — Synthetic banking transaction data from Kaggle.

| Property | Value |
|---|---|
| Total Transactions | 594,643 |
| Fraudulent | 7,200 (1.21%) |
| Legitimate | 587,443 (98.79%) |
| Imbalance Ratio | 1:81 |
| Simulation Period | ~6 months |
| Source | [kaggle.com/datasets/ealaxi/banksim1](https://www.kaggle.com/datasets/ealaxi/banksim1) |

---

## Experiments & Results

### E1 — Baseline ML Models

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.9491 | 0.8664 | 0.9059 | 0.9714 |
| SVM | 0.2617 | 0.3141 | 0.2855 | 0.4904 |
| Random Forest (Tuned) | 0.9949 | 1.0000 | 0.9975 | 0.9990 |
| XGBoost (Tuned) | 0.9432 | 1.0000 | 0.9708 | 0.9953 |

### E2 — Graph-Enhanced Hybrid Models

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| LR + Graph Features | 0.9682 | 0.9740 | 0.9711 | 0.9959 |
| RF + Graph Features | 0.9976 | 1.0000 | 0.9988 | **1.0000** |
| XGBoost + Graph Features | 0.9792 | 1.0000 | 0.9895 | 0.9993 |

### E3 — GraphSAGE Embedding Models

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| GraphSAGE + RF | 0.9963 | 1.0000 | 0.9981 | 0.9997 |
| GraphSAGE + MLP | 0.9933 | 1.0000 | 0.9966 | 0.9987 |

### E4 — Standalone GraphSAGE

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| GraphSAGE | 0.4344 | 0.9298 | 0.5922 | 0.7143 |

### E5 — Unsupervised Clustering

| Metric | Value |
|---|---|
| K-Means Purity Score | 0.9916 |
| K-Means Silhouette | 0.6315 |
| DBSCAN Noise Fraud Rate | 65.22% |

---

## Key Findings

- **Graph features improve every model** — LR recall jumped from 0.87 → 0.97, RF achieved perfect PR-AUC of 1.0
- **11 out of 49 merchants** have >50% fraud rate — invisible without graph analysis
- **K-Means Cluster 4** — 337 transactions, 99.7% fraud rate — pure fraud ring detected without labels
- **DBSCAN noise points** — 65.2% fraud rate vs 0.23% in normal clusters
- **GraphSAGE embeddings** as features for RF achieve near-perfect PR-AUC of 0.9997
- **Standalone GraphSAGE** has high recall (0.93) but low precision (0.43) — effective as a pre-filter, not standalone
- **Amount** has 0.49 correlation with fraud — strongest single tabular predictor
- **es_leisure** and **es_travel** categories have 80%+ fraud rates

---

## How to Run

### Prerequisites
- Python 3.10+
- Neo4j (optional — graph features fall back to pandas/networkx if unavailable)
- Node.js 16+ (only for the React dashboard)

### 1. Install Python dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Neo4j (optional)

If you have Neo4j running, set credentials via environment variables:

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your-password"
```

If Neo4j is not available, the graph module will fall back to networkx for betweenness centrality and community detection.

### 3. Run the full pipeline
```bash
python main.py
```

This executes all 8 steps in sequence. You can also run individual modules:

```bash
python preprocessing.py        # Step 1 — Data prep + SMOTE-ENN
python eda.py                  # Step 2 — EDA visualizations
python models.py               # Step 3 — E1: Baseline models (LR, SVM, RF, XGBoost)
python graph.py                # Step 4 — E2: Graph construction + features
python hybrid_models.py        # Step 5 — E2: Hybrid models
python graphsage.py            # Step 6 — E4: Standalone GraphSAGE
python embedding_models.py     # Step 7 — E3: Embedding models (RF + MLP)
python clustering.py           # Step 8 — E5: Clustering analysis
```

### 4. Run the React Dashboard
```bash
cd dashboard
npm install
npm start
```

Open `http://localhost:3000` to view:
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

| Feature | Description | Source |
|---|---|---|
| Degree Centrality | Number of transactions per customer/merchant | Neo4j / pandas |
| PageRank | Importance/centrality of each node in the network | Neo4j GDS |
| Betweenness Centrality | How often a node sits between other nodes | Neo4j GDS / networkx |
| Community Detection | Louvain cluster assignment for each node | Neo4j GDS / networkx |
| Merchant Fraud Rate | % of each merchant's transactions that are fraud | pandas (leak-safe: computed from training data only) |

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **SMOTE-ENN** over random oversampling | Combines synthetic minority oversampling with edited nearest neighbors cleanup — reduces noise from pure SMOTE |
| **PR-AUC** as primary metric | Precision-Recall AUC is more informative than ROC-AUC under severe class imbalance (1.2% fraud) |
| **Leak-safe merchant fraud rate** | Computed from training data only, then mapped to test set — prevents information leakage from test labels |
| **Neo4j with fallbacks** | Primary graph ops use Neo4j GDS for production scale; networkx/pandas fallbacks enable running without Neo4j |
| **Time-based split** | Uses the `step` column to split train/test chronologically, reflecting real-world fraud detection where you predict future fraud |
| **No accuracy metric** | With 1.2% fraud, a model predicting everything as legitimate achieves 98.8% accuracy — meaningless |

---

## Evaluation Metrics

| Metric | Why Used |
|---|---|
| **PR-AUC** | Primary metric — best for rare event detection under severe imbalance |
| **Recall** | Critical — missing fraud is costly |
| **Precision** | Ensures flagged fraud is actually fraud |
| **F1-Score** | Balance between Precision and Recall |
| **Purity Score** | Clustering quality — fraction of dominant class per cluster |
| **Silhouette Score** | Clustering separation quality |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| scikit-learn | ML models, preprocessing, evaluation |
| XGBoost | Gradient boosting baseline |
| imbalanced-learn | SMOTE-ENN class balancing |
| Neo4j + GDS | Graph database + graph algorithms |
| networkx | Betweenness centrality / community detection fallback |
| PyTorch | Deep learning framework |
| PyTorch Geometric | GraphSAGE implementation |
| pandas / numpy | Data manipulation |
| matplotlib / seaborn | Visualizations |
| React | Interactive results dashboard |

---

## References

1. Kaggle — BankSim1 Dataset: https://www.kaggle.com/datasets/ealaxi/banksim1
2. Neo4j Developer Guides: https://neo4j.com/developer/get-started/
3. Alloy — 2024 Fraud Stats: https://www.alloy.com/blog/2024-fraud-stats-for-banks-fintechs-and-credit-unions
4. Mathematics (MDPI), vol. 11, no. 13, Article 2862: https://www.mdpi.com/2227-7390/11/13/2862
5. Wang et al., "A Semi-Supervised Graph Attentive Network for Financial Fraud Detection," IEEE ICDM 2019. DOI: 10.1109/ICDM.2019.00070
