#!/usr/bin/env python3
"""
main.py
--------
End-to-end orchestration of the Banking Fraud Detection pipeline.

Usage:
    python main.py # currently, only download the BankSim dataset and perform EDA + balancing
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
    
    # TODO: Add parsers 
    
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

if __name__ == "__main__":
    main()
