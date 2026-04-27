# Architecture

## Pipeline Overview

The fraud detection pipeline runs 8 sequential steps, orchestrated by `CODE/main.py`:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CODE/main.py (orchestrator)                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Step 1: CODE/preprocessing.py ──→ Clean, encode, SMOTE-ENN, split     │
│  Step 2: CODE/eda.py ─────────────→ 7 EDA visualizations               │
│  Step 3: CODE/models.py ─────────→ E1: Baseline ML (LR, SVM, RF, XGB) │
│  Step 4: CODE/graph.py ──────────→ E2: Neo4j graph + feature extract  │
│  Step 5: CODE/hybrid_models.py ──→ E2: Graph-enhanced ML models        │
│  Step 6: CODE/graphsage.py ─────→ E4: Standalone GraphSAGE classifier  │
│  Step 7: CODE/embedding_models.py→ E3: GraphSAGE embeddings → RF + MLP │
│  Step 8: CODE/clustering.py ────→ E5: K-Means + DBSCAN fraud rings     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Project Layout

```
fraud-detection/
├── CODE/         ← All source code (Python modules + React dashboard)
├── DATA/         ← Raw input dataset (auto-downloaded, gitignored)
└── EVALUATIONS/  ← All generated outputs (plots, CSVs, metrics)
```

Every Python module computes a `PROJECT_ROOT` constant at import time:

```python
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
```

This resolves to the repository root regardless of which working directory you run the scripts from. `DATA_DIR` and `RESULTS_DIR` are derived from `PROJECT_ROOT`, so the code always reads from `<repo>/DATA/` and writes to `<repo>/EVALUATIONS/`.

## Data Flow

```
Kaggle (BankSim)
    │
    ▼
DATA/bs140513_032310.csv
    │
    ▼
CODE/preprocessing.py
    ├─ download_dataset()        ← kagglehub auto-download → DATA/
    ├─ load_data()               ← clean, drop nulls, type coercion
    ├─ encode_and_scale()        ← LabelEncoder + StandardScaler
    ├─ balance_data()            ← SMOTE-ENN (default) / oversample / undersample
    └─ time_based_split()        ← chronological train/test via "step" column
          │
          ▼
    ┌─────────────────┐
    │  X_train, X_test │
    │  y_train, y_test │
    └────────┬────────┘
             │
    ┌────────┴────────────────────────────────────────┐
    │                    │                             │
    ▼                    ▼                             ▼
 CODE/models.py     CODE/graph.py            CODE/graphsage.py
 (E1 Baseline)      (Graph Features)         (E4 Standalone GNN)
    │                    │                             │
    │                    ▼                             ▼
    │            CODE/hybrid_models.py        CODE/embedding_models.py
    │            (E2 Graph+Tabular)           (E3 Embeddings→ML)
    │                    │                             │
    └────────┬───────────┘                             │
             │                                         │
             ▼                                         │
      CODE/clustering.py  ◄────────────────────────────┘
      (E5 Unsupervised)
             │
             ▼
      EVALUATIONS/
      (CSVs, PNGs, comparison tables)
```

## Module Responsibilities

### `CODE/preprocessing.py`

Handles all data preparation from raw CSV to model-ready features.

| Function | Purpose |
|---|---|
| `download_dataset()` | Downloads BankSim from Kaggle via kagglehub if not present locally |
| `load_data(filepath)` | Loads CSV, drops nulls, coerces types, returns clean DataFrame |
| `show_class_imbalance(df)` | Prints fraud vs legitimate distribution |
| `encode_and_scale(df)` | LabelEncoder for categoricals, StandardScaler for numerics |
| `build_feature_matrix(df)` | Separates features (X) from label (y) |
| `balance_data(X, y, strategy)` | Applies SMOTE-ENN (default), random oversample, or undersample |
| `compare_balance_distributions(X, y)` | Prints class distribution for all 3 balancing strategies |
| `time_based_split(X, y, df)` | Splits by the `step` column chronologically (no future leakage) |
| `run_preprocessing(balance_strategy)` | Full pipeline: load → encode → balance → split → return dict |

### `CODE/eda.py`

Generates 7 visualization plots saved to `EVALUATIONS/eda/`.

| Function | Plot |
|---|---|
| `plot_class_distribution(df)` | Fraud vs legitimate bar chart |
| `plot_fraud_by_category(df)` | Fraud rate per merchant category |
| `plot_amount_distribution(df)` | Transaction amount histograms (fraud vs legit) |
| `plot_fraud_over_time(df)` | Temporal fraud rate trend |
| `plot_fraud_by_demographics(df)` | Fraud patterns by customer segments |
| `plot_merchant_fraud(df)` | Top merchants by fraud rate |
| `plot_correlation_heatmap(df)` | Feature correlation matrix |

### `CODE/models.py`

Trains and evaluates 4 baseline ML models with hyperparameter tuning.

| Function | Purpose |
|---|---|
| `evaluate_model(name, model, ...)` | Train → predict → compute Precision, Recall, F1, PR-AUC |
| `tune_random_forest(X_train, y_train)` | RandomizedSearchCV over RF hyperparameters |
| `tune_xgboost(X_train, y_train)` | RandomizedSearchCV over XGBoost hyperparameters |
| `tune_svm(X_train, y_train)` | RandomizedSearchCV over SVM hyperparameters (C, kernel, gamma) |
| `plot_pr_curves(results, y_test)` | Precision-Recall curves for all models |
| `plot_comparison(summary_df)` | Grouped bar chart comparing all metrics |
| `plot_confusion_matrices(results, y_test)` | Side-by-side confusion matrices |
| `run_baseline_models()` | Full E1 pipeline: preprocess → tune → train → evaluate → plot |

**Models:** Logistic Regression, SVM (balanced, probability=True), Random Forest, XGBoost.

### `CODE/graph.py`

Manages Neo4j graph construction and feature extraction.

| Method (FraudGraph class) | Purpose |
|---|---|
| `connect()` | Connects to Neo4j using env vars (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) |
| `clear_graph()` | Removes all existing nodes/edges |
| `build_graph(df)` | Creates Customer → Merchant edges from transactions |
| `extract_degree_centrality(df)` | Node degree as a feature |
| `extract_pagerank(df)` | PageRank scores per node |
| `extract_betweenness_centrality(df)` | Betweenness centrality (networkx fallback if no GDS) |
| `extract_community_ids(df)` | Louvain community detection (networkx fallback if no GDS) |
| `extract_merchant_fraud_rate(df, reference_df)` | Per-merchant fraud rate — **leak-safe**: uses `reference_df` (training data) to compute rates, fills unseen merchants with 0 |
| `extract_graph_features(df)` | Runs all feature extractors, saves to CSV |

### `CODE/hybrid_models.py`

Combines graph features with tabular features, retrains models, and compares against baselines.

| Function | Purpose |
|---|---|
| `add_graph_features(df)` | Merges graph features from `graph.py` onto the tabular DataFrame |
| `run_hybrid_models()` | Full E2 pipeline: preprocess → add graph features → train LR/RF/XGB → evaluate → compare vs baseline |
| `plot_pr_comparison(baseline, hybrid, y_test)` | Overlaid PR curves (baseline vs hybrid) |

### `CODE/graphsage.py`

Standalone GraphSAGE node classifier (E4).

| Function / Class | Purpose |
|---|---|
| `build_pyg_graph(df, sample_size)` | Converts transaction DataFrame to PyTorch Geometric `Data` object |
| `GraphSAGE` (class) | 2-layer GraphSAGE with dropout, trained end-to-end |
| `train_graphsage(data, epochs, lr, hidden)` | Trains the GNN, returns model + loss history |
| `evaluate_graphsage(model, data)` | Evaluates on test mask, returns metrics |
| `plot_final_comparison(graphsage_result)` | Compares GraphSAGE metrics against all previous experiments |
| `run_graphsage()` | Full E4 pipeline |

### `CODE/embedding_models.py`

Uses trained GraphSAGE to extract node embeddings, then trains traditional classifiers on them (E3).

| Function / Class | Purpose |
|---|---|
| `build_pyg_graph(df, sample_size)` | Same graph construction as graphsage.py |
| `GraphSAGEEncoder` (class) | Embedding extractor — GraphSAGE without classification head |
| `train_and_extract_embeddings(data, ...)` | Trains encoder, returns 32-dim embeddings per node |
| `map_embeddings_to_transactions(...)` | Maps node embeddings back to transaction-level features |
| `run_embedding_models()` | Full E3 pipeline: embed → train RF + MLP → evaluate |

### `CODE/clustering.py`

Unsupervised fraud ring detection (E5).

| Function | Purpose |
|---|---|
| `build_clustering_features(df)` | Extracts graph features and scales for clustering |
| `run_kmeans(X_scaled, y, n_clusters)` | K-Means clustering, analyzes fraud concentration per cluster |
| `run_dbscan(X_scaled, y, eps, min_samples)` | DBSCAN clustering, analyzes noise points vs fraud |
| `plot_clustering_results(...)` | 2x2 visualization: K-Means, DBSCAN, fraud overlay, cluster sizes |
| `compute_purity(labels, y)` | Purity score — fraction of dominant class per cluster |
| `run_clustering()` | Full E5 pipeline |

## Key Design Decisions

### 1. Data Leakage Prevention

`merchant_fraud_rate` is computed from training data only. The `extract_merchant_fraud_rate(df, reference_df)` method accepts a `reference_df` parameter — when set to the training set, fraud rates are computed from training labels and mapped to both train and test. Merchants unseen in training get a rate of 0.

### 2. Neo4j with Fallbacks

Graph feature extraction primarily uses Neo4j + GDS library for production-grade graph algorithms. When Neo4j is unavailable, betweenness centrality and community detection fall back to networkx. Degree centrality and PageRank fall back to pandas groupby operations. This allows the pipeline to run on any machine without Neo4j installed.

### 3. SMOTE-ENN Over Random Oversampling

With a 1:81 class imbalance, naive oversampling creates many duplicate fraud samples. SMOTE-ENN synthesizes new minority samples via interpolation (SMOTE), then uses Edited Nearest Neighbors to clean noisy boundary samples from both classes. This produces a more robust decision boundary.

### 4. Time-Based Train/Test Split

Instead of random stratified splitting, the pipeline uses the `step` column to split chronologically. Earlier time steps go to training, later steps to testing. This prevents temporal leakage — the model never sees future transactions during training, which mirrors real-world deployment.

### 5. Environment Variable Credentials

Neo4j credentials are read from environment variables (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`) with sensible defaults. No hardcoded passwords in source code.

### 6. PR-AUC as Primary Metric

Under severe class imbalance, ROC-AUC can be misleadingly high (a model that predicts all-legitimate gets high TNR). PR-AUC focuses on the minority class (fraud) and penalizes low precision or recall directly.

## Output Directory Structure

All pipeline outputs are written to `EVALUATIONS/` at the repo root:

```
EVALUATIONS/
├── baseline_results.csv             ← E1 metrics table
├── full_comparison.csv              ← All experiments combined
├── hybrid_results.csv               ← E2 metrics
├── e3_embedding_results.csv         ← E3 metrics
├── clustering/
│   ├── clustering_results.png       ← E5 cluster visualizations
│   └── clustering_summary.csv       ← E5 metrics
├── eda/
│   ├── 01_class_distribution.png
│   ├── 02_fraud_by_category.png
│   ├── 03_amount_distribution.png
│   ├── 04_fraud_over_time.png
│   ├── 05_fraud_by_demographics.png
│   ├── 06_merchant_fraud.png
│   └── 07_correlation_heatmap.png
├── pr_curves_baseline.png           ← E1 PR curves
├── pr_curves_comparison.png         ← E2 baseline vs hybrid
├── confusion_matrices.png           ← E1 confusion matrices
├── model_comparison.png             ← E1 bar chart
├── baseline_vs_hybrid.png           ← E2 bar chart
├── graphsage_loss.png               ← E4 training loss
├── graphsage_pr_curve.png           ← E4 PR curve
├── e3_embedding_pr_curves.png       ← E3 PR curves
└── full_comparison.png              ← All experiments bar chart
```

`graph_features.csv` is regenerated on demand by `CODE/graph.py` and is not committed to git (gitignored under both `EVALUATIONS/`).
