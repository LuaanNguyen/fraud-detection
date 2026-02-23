"""
preprocessing.py
---------------
- [ ] Explore, load, and clean BankSim dataset. 
- [ ] Fix the imbalance problem by either oversample or undersampling the data (normalization)
"""
import os

from kagglehub.gcs_upload import File 


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_FILE = os.path.join(DATA_DIR, "bs140513_032310.csv")

# ---------------------------------------------------------------------------
# 1. Dataset download helper
# ---------------------------------------------------------------------------
def download_dataset() -> str: 
    """Download BankSim dataset using kagglehub and return CSV path""" 
    if os.path.isfile(RAW_FILE):
        print(f"[INFO] Dataset already present at {RAW_FILE}")
        return RAW_FILE
    try: 
        import kagglehub
        
        print(f"[INFO] Downloading BankSim dataset via kagglehub...")
        path = kagglehub.dataset_download("ealaxi/banksim1")
        print(f"[INFO] Downloaded to: {path}")
        
        # Locate the CSV file inside the download directory 
        import glob 
        
        candidates = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True) 
        if not candidates:
            raise FileNotFoundError(
                f"No CSV files found in the downloaded path: {path}"
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