#!/usr/bin/env python3
"""
main.py
--------
End-to-end orchestration of the Banking Fraud Detection pipeline.

Usage:
    python main.py                        # Neo4j mode (requires running Neo4j + GDS)
    python main.py --no-neo4j             # Simulated graph features (no Neo4j needed)
    python main.py --data path/to/csv     # Custom CSV path
    python main.py --balance undersample  # Use undersampling instead

Environment variables (for Neo4j mode):
    NEO4J_URI       default: bolt://localhost:7687
    NEO4J_USER      default: neo4j
    NEO4J_PASSWORD  default: password
"""

import argparse
import sys
import warnings

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots

warnings.filterwarnings("ignore")

from preprocessing import run_preprocessing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Banking Fraud Detection — Baseline vs Graph-Enhanced Models"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to BankSim CSV file (default: auto-download)",
    )
    parser.add_argument(
        "--balance",
        type=str,
        choices=["oversample", "undersample"],
        default="oversample",
        help="Class balancing strategy (default: oversample)",
    )
    parser.add_argument(
        "--no-neo4j",
        action="store_true",
        default=False,
        help="Use simulated graph features instead of Neo4j",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 70)
    print("   BANKING FRAUD DETECTION PIPELINE")
    print("   Baseline (Tabular) vs Graph-Enhanced Models")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Step 1: Preprocessing
    # ---------------------------------------------------------------
    prep = run_preprocessing(
        filepath=args.data,
        balance_strategy=args.balance,
    )
    
    print(prep)


if __name__ == "__main__":
    main()
