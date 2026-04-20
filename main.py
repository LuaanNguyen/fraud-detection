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
from eda import run_eda


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
        choices=["smote_enn", "oversample", "undersample"],
        default="smote_enn",
        help="Class balancing strategy (default: smote_enn)",
    )

    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip the EDA step (plots already generated)",
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

    # ---------------------------------------------------------------
    # Step 2: Exploratory Data Analysis
    # ---------------------------------------------------------------
    if not args.skip_eda:
        run_eda(prep["df_raw"])

if __name__ == "__main__":
    main()
