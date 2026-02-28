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

### 1. Install dependencies + venv

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the preprocessing pipeline

The `main.py` entry point orchestrates the full pipeline. It downloads the [BankSim dataset](https://www.kaggle.com/datasets/ealaxi/banksim1) automatically if it is not already present in `data/`.

```bash
python main.py
```

This will:

1. Download the BankSim dataset (via `kagglehub`) and cache it locally in `data/`
2. Load and clean the CSV (handle missing values, drop constant columns)
3. Encode categorical features and scale numeric features
4. Display class-imbalance statistics
5. Compare oversampling and undersampling distributions
6. Produce a stratified train/test split on the balanced data

> **Note:** `main.py` argument parsing is still under development. If you encounter issues, you can run the preprocessing pipeline directly in a Python shell:
>
> ```python
> from preprocessing import run_preprocessing
> result = run_preprocessing()  # downloads data, defaults to oversampling
> ```

### Manual dataset setup (optional)

If you prefer not to use the Kaggle API, download the CSV manually from [BankSim1 on Kaggle](https://www.kaggle.com/datasets/ealaxi/banksim1) and place it at:

```
data/bs140513_032310.csv
```

## Important documents:

- [Project Proposal](https://docs.google.com/document/d/1cGvCQ9Vi4sMQLHlTNQHU1j5jcXeqkZ2ZvuOZDr7Ufe8/edit?tab=t.0)
- [Canvas]()
