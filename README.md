# Fraud Detection Using Graph Databases

## Student Information

| Name                           |      Email       |     ASU ID |
| :----------------------------- | :--------------: | ---------: |
| Luan Nguyen                    | ltnguy58@asu.edu | 1225177265 |
| Dhanush Mugajji Shambulingappa | dmugajji@asu.edu | 1237526292 |
| Chandra Shekhar Pavuluri       | cpavulur@asu.edu | 1236078196 |
| Adrian Zhang                   | awzhang1@asu.edu | 1224664415 |
| Shashikant Nanda               | snanda5@asu.edu  | 1235508926 |
| Chitwandeep Kaur Palne         |  cpalne@asu.edu  | 1231519031 |

## How to Run

### Prerequisites

- Python 3.10+
- (Optional) Neo4j 5.x with GDS plugin for full graph features — the pipeline falls back to pandas-based proxies if Neo4j is unavailable

### 1. Set up environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python main.py
```

This runs preprocessing (with SMOTE-ENN balancing) and EDA. The BankSim dataset is auto-downloaded on first run.

**Options:**

```bash
python main.py --balance oversample    # use random oversampling instead of SMOTE-ENN
python main.py --balance undersample   # use random undersampling
python main.py --skip-eda              # skip EDA plot generation
python main.py --data path/to/file.csv # use a custom CSV path
```

### 3. Run individual modules

Each module can also be run standalone:

```bash
python eda.py              # EDA only (plots saved to output/eda/)
python models.py           # Baseline models with hyperparameter tuning (LR, SVM, RF, XGBoost)
python graph.py            # Neo4j graph construction + feature extraction
python hybrid_models.py   # Graph-enhanced models vs baseline comparison
```

### Manual dataset setup (optional)

If you prefer not to use the Kaggle API, download the CSV manually from [BankSim1 on Kaggle](https://www.kaggle.com/datasets/ealaxi/banksim1) and place it at:

```
data/bs140513_032310.csv
```

### Neo4j setup (optional)

For full graph features (PageRank, Betweenness Centrality, Louvain community detection), install Neo4j and set environment variables:

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=yourpassword
```

Without Neo4j, the pipeline uses pandas-based approximations automatically.

## Project Structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full module dependency graph, data flow, and design decisions.

```
main.py              → pipeline orchestrator
preprocessing.py     → data loading, cleaning, encoding, balancing, splitting
eda.py               → exploratory data analysis (7 plot types)
models.py            → baseline ML models + GridSearchCV tuning
graph.py             → Neo4j graph pipeline + feature extraction
hybrid_models.py     → graph-enhanced model comparison
eval_utils.py        → shared evaluation function
```

## Important Documents

- [Project Proposal](https://docs.google.com/document/d/1cGvCQ9Vi4sMQLHlTNQHU1j5jcXeqkZ2ZvuOZDr7Ufe8/edit?tab=t.0)
- [Architecture](ARCHITECTURE.md)
- [Canvas]()
