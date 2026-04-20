# Architecture

## Overview

This project compares traditional tabular ML models against graph-enhanced models for fraud detection on the BankSim dataset. The codebase is organized as a pipeline with clear module boundaries.

## Data Flow

```
BankSim CSV (Kaggle)
       │
       ▼
  preprocessing.py    ←── load, clean, encode, scale, SMOTE-ENN balance, train/test split
       │
       ├──► eda.py              ←── exploratory plots (saved to output/eda/)
       │
       ├──► models.py           ←── Phase 2: baseline ML (LR, SVM, RF, XGBoost + tuning)
       │
       ├──► graph.py            ←── Phase 3: Neo4j graph construction + feature extraction
       │       │                     (degree, PageRank, betweenness, community, fraud rate)
       │       ▼
       └──► hybrid_models.py    ←── Phase 4: retrain ML on tabular + graph features
               │
               ▼
          eval_utils.py         ←── shared evaluate_model() used by models.py & hybrid_models.py
```

## Module Responsibilities

| File | Purpose | Can run standalone? |
|---|---|---|
| `main.py` | Pipeline orchestrator. Runs preprocessing + EDA. | Yes: `python main.py` |
| `preprocessing.py` | Data loading (auto-download via kagglehub), cleaning, encoding, scaling, class balancing (SMOTE-ENN / oversample / undersample), stratified train/test split. | No (library module) |
| `eda.py` | Exploratory data analysis — 7 plot types saved to `output/eda/`. | Yes: `python eda.py` |
| `models.py` | Baseline models with hyperparameter tuning via GridSearchCV. Outputs results + plots to `results/`. | Yes: `python models.py` |
| `graph.py` | Neo4j graph pipeline — builds customer-merchant graph, extracts graph features. Falls back to pandas-based proxies when GDS is unavailable. | Yes: `python graph.py` |
| `hybrid_models.py` | Graph-enhanced models — merges tabular + graph features, retrains, compares against baseline. | Yes: `python hybrid_models.py` |
| `eval_utils.py` | Shared `evaluate_model()` function (train, predict, compute metrics, print report). | No (library module) |

## Key Design Decisions

### Neo4j fallbacks
`graph.py` tries Neo4j GDS for PageRank, Betweenness Centrality, and Louvain community detection. If GDS is not installed, it falls back to pandas-based proxies so the pipeline can run without Neo4j for development/testing.

### Data leakage prevention
`merchant_fraud_rate` in `hybrid_models.py` is computed from **training data only** and mapped to the test set. Unseen merchants default to rate 0. This was a fix applied to the original implementation.

### Class balancing
The default strategy is **SMOTE-ENN** (per the project proposal), which oversamples the minority class with SMOTE then cleans noisy samples with Edited Nearest Neighbours. Random oversample and undersample are available as alternatives via the `--balance` flag.

## Directory Structure

```
fraud-detection/
├── main.py                  # Entry point
├── preprocessing.py         # Data pipeline
├── eda.py                   # Exploratory data analysis
├── models.py                # Baseline ML models + tuning
├── graph.py                 # Neo4j graph pipeline
├── hybrid_models.py         # Graph-enhanced models
├── eval_utils.py            # Shared evaluation utilities
├── requirements.txt         # Python dependencies
├── data/                    # BankSim CSV (gitignored, auto-downloaded)
├── output/eda/              # EDA plots (gitignored, regenerated)
├── results/                 # Model results, plots, CSVs
└── ARCHITECTURE.md          # This file
```

## Where Phase 5 and 6 Plug In

### Phase 5: GraphSAGE + MLP
- **Input**: the graph structure from `graph.py` (customer/merchant nodes, transaction edges) + the feature matrix from `preprocessing.py`
- **New file(s)**: e.g. `graphsage.py` — build a DGL graph, train GraphSAGE, extract node embeddings, feed into MLP and XGBoost
- **Reuse**: `eval_utils.evaluate_model()` for consistent metric reporting, `preprocessing.load_data()` for data loading

### Phase 6: Comparative Evaluation + Clustering
- **Input**: results from `models.py` (baseline), `hybrid_models.py` (graph-enhanced), and the Phase 5 embedding models
- **New file(s)**: e.g. `comparison.py` — load all result CSVs, produce unified comparison tables/charts. `clustering.py` — K-Means/DBSCAN on graph features and embeddings, purity metric, fraud density analysis
- **Reuse**: graph features from `graph.py`, embeddings from Phase 5, `eval_utils.evaluate_model()` if cluster membership is used as a feature
