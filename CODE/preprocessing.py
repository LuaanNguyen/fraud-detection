"""
preprocessing.py
-----------------
Data loading, cleaning, encoding, normalization, class imbalance handling,
and train/test splitting for the BankSim fraud-detection dataset.
"""

import os
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "DATA")
RAW_FILE = os.path.join(DATA_DIR, "bs140513_032310.csv")

LABEL_COL = "fraud"
CUSTOMER_COL = "customer"
MERCHANT_COL = "merchant"

CATEGORICAL_COLS = ["age", "gender", "category"]
NUMERIC_COLS = ["amount"]

# Columns that are IDs / not features / low-information
ID_COLS = ["customer", "merchant", "step", "zipcodeOri", "zipMerchant"]

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Dataset download helper
# ---------------------------------------------------------------------------
def download_dataset() -> str:
    """Download BankSim dataset using kagglehub and return CSV path."""
    if os.path.isfile(RAW_FILE):
        print(f"[INFO] Dataset already present at {RAW_FILE}")
        return RAW_FILE

    try:
        import kagglehub
        print("[INFO] Downloading BankSim dataset via kagglehub …")
        path = kagglehub.dataset_download("ealaxi/banksim1")
        print(f"[INFO] Downloaded to: {path}")

        import glob
        candidates = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
        if not candidates:
            raise FileNotFoundError(
                f"No CSV files found in downloaded path: {path}"
            )
        src = candidates[0]

        os.makedirs(DATA_DIR, exist_ok=True)
        import shutil
        shutil.copy2(src, RAW_FILE)
        print(f"[INFO] Copied dataset to {RAW_FILE}")
        return RAW_FILE

    except Exception as e:
        raise RuntimeError(
            f"Failed to download dataset: {e}\n"
            f"Please manually download from "
            f"https://www.kaggle.com/datasets/ealaxi/banksim1 "
            f"and place the CSV at {RAW_FILE}"
        ) from e


# ---------------------------------------------------------------------------
# 2. Load & basic clean
# ---------------------------------------------------------------------------
def load_data(filepath: str | None = None) -> pd.DataFrame:
    """Load the BankSim CSV and perform basic cleaning."""
    if filepath is None:
        filepath = download_dataset()

    print(f"\n{'='*60}")
    print("LOADING DATA")
    print(f"{'='*60}")

    df = pd.read_csv(filepath, quotechar="'")

    # Strip leading/trailing whitespace from column names
    df.columns = df.columns.str.strip().str.replace("'", "")

    # Ensure fraud column is int
    if LABEL_COL in df.columns:
        df[LABEL_COL] = df[LABEL_COL].astype(int)

    # Drop single-value / low-info columns
    for col in ["zipcodeOri", "zipMerchant"]:
        if col in df.columns and df[col].nunique() <= 1:
            print(f"  Dropping constant column: {col} (unique={df[col].nunique()})")
            df.drop(columns=[col], inplace=True)

    print(f"  Shape           : {df.shape}")
    print(f"  Columns         : {list(df.columns)}")
    print(f"  Dtypes:\n{df.dtypes.to_string()}\n")

    # --- Handle missing values ---
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("  Missing values detected:")
        print(missing[missing > 0].to_string())
        df.dropna(subset=[LABEL_COL], inplace=True)
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype == "object":
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
        print("  → Missing values handled.\n")
    else:
        print("  No missing values found.\n")

    return df


# ---------------------------------------------------------------------------
# 3. Class-imbalance statistics
# ---------------------------------------------------------------------------
def show_class_imbalance(df: pd.DataFrame) -> None:
    """Print class distribution statistics."""
    print(f"\n{'='*60}")
    print("CLASS IMBALANCE")
    print(f"{'='*60}")
    counts = df[LABEL_COL].value_counts()
    pcts = df[LABEL_COL].value_counts(normalize=True) * 100
    for label in counts.index:
        tag = "Fraud" if label == 1 else "Legitimate"
        print(f"  {tag} ({label}): {counts[label]:>8,}  ({pcts[label]:.2f}%)")
    print(f"  Imbalance ratio : 1:{counts[0] // max(counts[1], 1)}")


# ---------------------------------------------------------------------------
# 4. Feature engineering (encode + scale)
# ---------------------------------------------------------------------------
def encode_and_scale(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder], StandardScaler]:
    """Label-encode categoricals and standard-scale numerics."""
    df_enc = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    # Encode categorical columns
    for col in CATEGORICAL_COLS:
        if col in df_enc.columns:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            encoders[col] = le

    # Scale numeric columns
    scaler = StandardScaler()
    num_cols_present = [c for c in NUMERIC_COLS if c in df_enc.columns]
    if num_cols_present:
        df_enc[num_cols_present] = scaler.fit_transform(df_enc[num_cols_present])

    return df_enc, encoders, scaler


# ---------------------------------------------------------------------------
# 5. Build feature matrix
# ---------------------------------------------------------------------------
def build_feature_matrix(
    df_encoded: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return X (features) and y (label)."""
    feature_cols = [
        c for c in df_encoded.columns if c not in ID_COLS + [LABEL_COL]
    ]
    X = df_encoded[feature_cols].copy()
    y = df_encoded[LABEL_COL].copy()
    return X, y


# ---------------------------------------------------------------------------
# 6. Handle class imbalance
# ---------------------------------------------------------------------------
def balance_data(
    X: pd.DataFrame, y: pd.Series, strategy: str = "smoteenn"
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE-ENN, random over- or under-sampling.

    Parameters
    ----------
    strategy : 'smoteenn' | 'oversample' | 'undersample'
    """
    if strategy == "smoteenn":
        print("  [INFO] Applying SMOTE-ENN balancing...")
        sampler = SMOTEENN(random_state=RANDOM_STATE)
    elif strategy == "oversample":
        sampler = RandomOverSampler(random_state=RANDOM_STATE)
    elif strategy == "undersample":
        sampler = RandomUnderSampler(random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res


def compare_balance_distributions(
    y_orig: pd.Series,
    y_over: pd.Series,
    y_under: pd.Series,
) -> None:
    """Print before/after class distributions."""
    print(f"\n{'='*60}")
    print("CLASS DISTRIBUTION COMPARISON")
    print(f"{'='*60}")
    for tag, y in [
        ("Original", y_orig),
        ("Oversampled", y_over),
        ("Undersampled", y_under),
    ]:
        counts = y.value_counts().sort_index()
        total = len(y)
        print(
            f"  {tag:<15s} | "
            f"Legit: {counts.get(0, 0):>8,} ({counts.get(0, 0)/total*100:.1f}%) | "
            f"Fraud: {counts.get(1, 0):>8,} ({counts.get(1, 0)/total*100:.1f}%)"
        )


# ---------------------------------------------------------------------------
# 7. Stratified train/test split
# ---------------------------------------------------------------------------
def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )


# ---------------------------------------------------------------------------
# 8. Time-based train/test split (prevents data leakage)
# ---------------------------------------------------------------------------
def time_based_split(
    df_encoded: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data based on the step column (time) to prevent data leakage.
    Earlier steps = training, later steps = testing.
    """
    if "step" not in df_encoded.columns:
        print("[WARNING] step column not found, falling back to stratified split")
        X, y = build_feature_matrix(df_encoded)
        return stratified_split(X, y, test_size)

    df_sorted = df_encoded.sort_values("step").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))

    train_df = df_sorted.iloc[:split_idx]
    test_df  = df_sorted.iloc[split_idx:]

    feature_cols = [
        c for c in df_sorted.columns if c not in ID_COLS + [LABEL_COL]
    ]

    X_train = train_df[feature_cols]
    y_train = train_df[LABEL_COL]
    X_test  = test_df[feature_cols]
    y_test  = test_df[LABEL_COL]

    print(f"\n  [INFO] Time-based split on step column")
    print(f"  Train steps  : {train_df['step'].min()} → {train_df['step'].max()}")
    print(f"  Test steps   : {test_df['step'].min()} → {test_df['step'].max()}")
    print(f"  Training set : {X_train.shape[0]:,} samples")
    print(f"  Test set     : {X_test.shape[0]:,} samples")
    print(f"  Train fraud  : {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
    print(f"  Test fraud   : {y_test.sum():,} ({y_test.mean()*100:.2f}%)")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 9. Full preprocessing pipeline (convenience)
# ---------------------------------------------------------------------------
def run_preprocessing(
    filepath: str | None = None,
    balance_strategy: str = "smoteenn",
) -> dict:
    """Run the full preprocessing pipeline and return a dict of artefacts."""

    # Load
    df_raw = load_data(filepath)
    show_class_imbalance(df_raw)

    # Encode & scale
    df_enc, encoders, scaler = encode_and_scale(df_raw)

    # Feature matrix
    X, y = build_feature_matrix(df_enc)

    # Balance comparisons
    X_over, y_over = balance_data(X, y, strategy="oversample")
    X_under, y_under = balance_data(X, y, strategy="undersample")
    compare_balance_distributions(y, y_over, y_under)

    # SMOTE-ENN balancing
    print(f"\n{'='*60}")
    print("SMOTE-ENN BALANCING")
    print(f"{'='*60}")
    X_bal, y_bal = balance_data(X, y, strategy=balance_strategy)
    print(f"  Balanced set : {X_bal.shape[0]:,} samples")
    print(f"  Fraud        : {y_bal.sum():,} ({y_bal.mean()*100:.2f}%)")
    print(f"  Legit        : {(y_bal==0).sum():,} ({(y_bal==0).mean()*100:.2f}%)")

    # Stratified split on balanced data
    X_train, X_test, y_train, y_test = stratified_split(X_bal, y_bal)
    print(f"\n  Training set : {X_train.shape[0]:,} samples")
    print(f"  Test set     : {X_test.shape[0]:,} samples")

    # Time-based split on raw encoded data (leakage-free)
    print(f"\n{'='*60}")
    print("TIME-BASED SPLIT (Leakage-Free)")
    print(f"{'='*60}")
    X_train_t, X_test_t, y_train_t, y_test_t = time_based_split(df_enc)

    return {
        "df_raw"        : df_raw,
        "df_encoded"    : df_enc,
        "encoders"      : encoders,
        "scaler"        : scaler,
        "X"             : X,
        "y"             : y,
        "X_train"       : X_train,
        "X_test"        : X_test,
        "y_train"       : y_train,
        "y_test"        : y_test,
        "X_train_time"  : X_train_t,
        "X_test_time"   : X_test_t,
        "y_train_time"  : y_train_t,
        "y_test_time"   : y_test_t,
        "feature_names" : list(X.columns),
    }