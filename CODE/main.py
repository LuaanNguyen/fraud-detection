#!/usr/bin/env python3
"""
main.py
--------
End-to-end orchestration of the Banking Fraud Detection pipeline.
Runs all experiments in sequence:
  Step 1 — Preprocessing + SMOTE-ENN + Time-based split
  Step 2 — EDA visualizations
  Step 3 — E1: Baseline ML models
  Step 4 — E2: Graph construction + feature extraction
  Step 5 — E2: Hybrid models
  Step 6 — E4: Standalone GraphSAGE
  Step 7 — E3: GraphSAGE embeddings → RF + MLP
  Step 8 — E5: Clustering analysis
"""

import warnings
warnings.filterwarnings("ignore")

def main():
    print("\n" + "=" * 70)
    print("   BANKING FRAUD DETECTION PIPELINE")
    print("   Baseline (Tabular) vs Graph-Enhanced vs GNN Models")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Step 1: Preprocessing
    # ---------------------------------------------------------------
    print("\n[STEP 1] Preprocessing + SMOTE-ENN + Time-based Split")
    from preprocessing import run_preprocessing
    prep = run_preprocessing(balance_strategy="smoteenn")
    print(" Preprocessing complete")

    # ---------------------------------------------------------------
    # Step 2: EDA
    # ---------------------------------------------------------------
    print("\n[STEP 2] Exploratory Data Analysis")
    from eda import run_eda
    run_eda()
    print(" EDA complete")

    # ---------------------------------------------------------------
    # Step 3: E1 Baseline Models
    # ---------------------------------------------------------------
    print("\n[STEP 3] E1 — Baseline ML Models")
    from models import run_baseline_models
    baseline_results = run_baseline_models()
    print(" Baseline models complete")

    # ---------------------------------------------------------------
    # Step 4: E2 Graph Construction
    # ---------------------------------------------------------------
    print("\n[STEP 4] E2 — Graph Construction + Feature Extraction")
    from graph import run_graph_pipeline
    run_graph_pipeline()
    print(" Graph pipeline complete")

    # ---------------------------------------------------------------
    # Step 5: E2 Hybrid Models
    # ---------------------------------------------------------------
    print("\n[STEP 5] E2 — Hybrid Models (Graph + Tabular)")
    from hybrid_models import run_hybrid_models
    run_hybrid_models()
    print(" Hybrid models complete")

    # ---------------------------------------------------------------
    # Step 6: E4 Standalone GraphSAGE
    # ---------------------------------------------------------------
    print("\n[STEP 6] E4 — Standalone GraphSAGE")
    from graphsage import run_graphsage
    run_graphsage()
    print(" GraphSAGE complete")

    # ---------------------------------------------------------------
    # Step 7: E3 Embedding Models
    # ---------------------------------------------------------------
    print("\n[STEP 7] E3 — GraphSAGE Embeddings → RF + MLP")
    from embedding_models import run_embedding_models
    run_embedding_models()
    print(" Embedding models complete")

    # ---------------------------------------------------------------
    # Step 8: E5 Clustering
    # ---------------------------------------------------------------
    print("\n[STEP 8] E5 — Unsupervised Clustering")
    from clustering import run_clustering
    run_clustering()
    print(" Clustering complete")

    # ---------------------------------------------------------------
    # Done
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("   PIPELINE COMPLETE")
    print("   All results saved to EVALUATIONS/")
    print("   GitHub: https://github.com/LuaanNguyen/fraud-detection")
    print("=" * 70)


if __name__ == "__main__":
    main()