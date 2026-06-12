# %% [markdown]
# # Monsoon CreditTech - Credit Risk Modelling
#
# I kept the notebook in the order I used while working:
# load the raw files, check the data, build borrower-level features, validate with folds,
# and then write the submission file.
#
# Metric used for model selection: ROC-AUC.
#
# Data note: the account and enquiry files come as nested JSON lists, so I flatten them first.
# In Colab, keep the `data/` folder under `/content/data`; locally this also runs from the project root.

# %%
# 01. Imports and Colab setup

import importlib.util
import os
import random
import re
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def install_if_missing(import_name: str, pip_name: str | None = None) -> None:
    """Small Colab helper. Locally it simply skips packages that are already installed."""
    if importlib.util.find_spec(import_name) is None:
        package = pip_name or import_name
        print(f"Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


for import_name, pip_name in [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("lightgbm", "lightgbm"),
    ("xgboost", "xgboost"),
    ("shap", "shap"),
]:
    install_if_missing(import_name, pip_name)

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb

from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from IPython.display import display
except Exception:
    display = print

sns.set_theme(style="whitegrid", palette="viridis")
pd.set_option("display.max_columns", 250)
pd.set_option("display.max_rows", 120)

# %%
# 02. Run settings

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

# Change this for the exact required filename:
# final_submission_'your_first_name_your_last_name'.csv
YOUR_FIRST_LAST_NAME = "lakshya"

# quick: use this while checking the notebook end-to-end
# strong: default run I would submit from Colab
# max: slower, only worth it if there is enough runtime left
SEARCH_LEVEL = "strong"

SEARCH_CONFIGS = {
    "quick": {"n_folds": 5, "lr_iter": 8, "lgb_iter": 4, "xgb_iter": 4, "early_stop": 50},
    "strong": {"n_folds": 5, "lr_iter": 25, "lgb_iter": 12, "xgb_iter": 12, "early_stop": 100},
    "max": {"n_folds": 5, "lr_iter": 50, "lgb_iter": 30, "xgb_iter": 30, "early_stop": 150},
}

CFG = SEARCH_CONFIGS[SEARCH_LEVEL]
N_FOLDS = CFG["n_folds"]
EARLY_STOPPING_ROUNDS = CFG["early_stop"]

TOP_N_CREDIT_TYPES = 12
TOP_N_ENQUIRY_TYPES = 17
RECENCY_WINDOWS = [30, 90, 180, 365, 730, 1095]
DPD_WINDOWS = [3, 6, 12, 24]
CLIP_QUANTILES = (0.001, 0.999)
SHAP_SAMPLE_SIZE = 2500

print(f"Run mode: {SEARCH_LEVEL}")
print(f"Folds: {N_FOLDS} | LR trials: {CFG['lr_iter']} | LGB trials: {CFG['lgb_iter']} | XGB trials: {CFG['xgb_iter']}")

# %%
# 03. File paths


def find_path(*relative_candidates: str) -> Path:
    """Handle both Colab paths and the local project folder."""
    roots = [Path.cwd(), Path("/content"), Path("/content/drive/MyDrive")]
    for root in roots:
        for rel in relative_candidates:
            path = root / rel
            if path.exists():
                return path
    searched = [str(root / rel) for root in roots for rel in relative_candidates]
    raise FileNotFoundError("Could not find any of:\n" + "\n".join(searched[:20]))


TRAIN_FLAG_PATH = find_path("data/train/train_flag.csv", "train/train_flag.csv", "train_flag.csv")
TEST_FLAG_PATH = find_path("data/test/test_flag.csv", "test/test_flag.csv", "test_flag.csv")
ACCOUNTS_TRAIN_PATH = find_path("data/train/accounts_data_train.json", "train/accounts_data_train.json", "accounts_data_train.json")
ENQUIRY_TRAIN_PATH = find_path("data/train/enquiry_data_train.json", "train/enquiry_data_train.json", "enquiry_data_train.json")
ACCOUNTS_TEST_PATH = find_path("data/test/accounts_data_test.json", "test/accounts_data_test.json", "accounts_data_test.json")
ENQUIRY_TEST_PATH = find_path("data/test/enquiry_data_test.json", "test/enquiry_data_test.json", "enquiry_data_test.json")
SAMPLE_SUBMISSION_PATH = find_path(
    "data/final_submission/sample_submission.csv",
    "final_submission/sample_submission.csv",
    "sample_submission.csv",
)
OUTPUT_DIR = SAMPLE_SUBMISSION_PATH.parent

for label, path in {
    "train_flag": TRAIN_FLAG_PATH,
    "test_flag": TEST_FLAG_PATH,
    "accounts_train": ACCOUNTS_TRAIN_PATH,
    "enquiry_train": ENQUIRY_TRAIN_PATH,
    "accounts_test": ACCOUNTS_TEST_PATH,
    "enquiry_test": ENQUIRY_TEST_PATH,
    "sample_submission": SAMPLE_SUBMISSION_PATH,
}.items():
    print(f"{label:18s}: {path}")

# %%
# 04. Load the raw files


def flatten_nested_json(path: Path, name: str) -> pd.DataFrame:
    """Each row contains several dictionaries; this converts them to a normal long table."""
    print(f"Loading {name}: {path}")
    raw = pd.read_json(path)
    records = []
    for row in raw.itertuples(index=False, name=None):
        records.extend(item for item in row if isinstance(item, dict))
    out = pd.DataFrame.from_records(records)
    print(f"{name:16s} raw={raw.shape} flattened={out.shape}")
    return out


train_flag = pd.read_csv(TRAIN_FLAG_PATH)
test_flag = pd.read_csv(TEST_FLAG_PATH)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

accounts_train_raw = flatten_nested_json(ACCOUNTS_TRAIN_PATH, "accounts_train")
enquiries_train_raw = flatten_nested_json(ENQUIRY_TRAIN_PATH, "enquiries_train")
accounts_test_raw = flatten_nested_json(ACCOUNTS_TEST_PATH, "accounts_test")
enquiries_test_raw = flatten_nested_json(ENQUIRY_TEST_PATH, "enquiries_test")

print("\nFlag shapes:")
print("train_flag:", train_flag.shape)
print("test_flag :", test_flag.shape)
print("sample    :", sample_submission.shape)

display(train_flag.head())
display(accounts_train_raw.head())
display(enquiries_train_raw.head())

# %%
# 05. Basic cleaning


def clean_accounts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["credit_type"] = df["credit_type"].fillna("Unknown").astype(str)
    df["loan_amount"] = pd.to_numeric(df["loan_amount"], errors="coerce")
    df["amount_overdue"] = pd.to_numeric(df["amount_overdue"], errors="coerce")
    df["loan_amount_missing"] = df["loan_amount"].isna().astype(np.int8)
    df["amount_overdue_missing"] = df["amount_overdue"].isna().astype(np.int8)
    df["loan_amount"] = df["loan_amount"].fillna(0).clip(lower=0)
    df["amount_overdue"] = df["amount_overdue"].fillna(0).clip(lower=0)

    df["open_date"] = pd.to_datetime(df["open_date"], errors="coerce")
    df["closed_date"] = pd.to_datetime(df["closed_date"], errors="coerce")
    invalid_close = df["closed_date"].notna() & df["open_date"].notna() & (df["closed_date"] < df["open_date"])
    df["closed_date_before_open_flag"] = invalid_close.astype(np.int8)
    df.loc[invalid_close, "closed_date"] = pd.NaT

    df["payment_hist_string"] = df["payment_hist_string"].fillna("").astype(str)
    bad_hist_tokens = df["payment_hist_string"].str.lower().isin(["nan", "none", "nat"])
    df.loc[bad_hist_tokens, "payment_hist_string"] = ""
    return df


def clean_enquiries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["enquiry_type"] = df["enquiry_type"].fillna("Unknown").astype(str)
    df["enquiry_amt"] = pd.to_numeric(df["enquiry_amt"], errors="coerce")
    df["enquiry_amt_missing"] = df["enquiry_amt"].isna().astype(np.int8)
    df["enquiry_amt"] = df["enquiry_amt"].fillna(0).clip(lower=0)
    df["enquiry_date"] = pd.to_datetime(df["enquiry_date"], errors="coerce")
    return df


accounts_train = clean_accounts(accounts_train_raw)
accounts_test = clean_accounts(accounts_test_raw)
enquiries_train = clean_enquiries(enquiries_train_raw)
enquiries_test = clean_enquiries(enquiries_test_raw)

train_flag["NAME_CONTRACT_TYPE"] = train_flag["NAME_CONTRACT_TYPE"].fillna("Unknown").astype(str)
test_flag["NAME_CONTRACT_TYPE"] = test_flag["NAME_CONTRACT_TYPE"].fillna("Unknown").astype(str)

REFERENCE_DATE = max(
    accounts_train["open_date"].max(),
    accounts_train["closed_date"].max(),
    enquiries_train["enquiry_date"].max(),
)
print("Train-only reference date:", REFERENCE_DATE.date())

# %%
# 06. First data checks


def describe_frame(df: pd.DataFrame, name: str, key_col: str = "uid") -> None:
    print(f"\n{'=' * 90}")
    print(name)
    print(f"{'=' * 90}")
    print("Shape:", df.shape)
    if key_col in df.columns:
        print(f"Unique {key_col}:", df[key_col].nunique())
    print("Duplicate rows:", int(df.duplicated().sum()))
    print("\nDtypes:")
    print(df.dtypes)
    print("\nMissing %:")
    print(df.isna().mean().mul(100).sort_values(ascending=False).round(3))
    print("\nNumeric describe:")
    display(df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        print("\nCategorical describe:")
        display(df[cat_cols].describe().T)


describe_frame(train_flag, "Train flag")
describe_frame(test_flag, "Test flag")
describe_frame(accounts_train, "Accounts train long")
describe_frame(enquiries_train, "Enquiries train long")

print("\nTarget distribution:")
display(train_flag["TARGET"].value_counts(normalize=True).rename("ratio").mul(100).round(3))
print("\nAccount UID coverage in train:", accounts_train["uid"].nunique(), "of", train_flag["uid"].nunique())
print("Enquiry UID coverage in train:", enquiries_train["uid"].nunique(), "of", train_flag["uid"].nunique())

# %%
# 07. Plotting helpers


def plot_target_distribution(flag_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    order = [0, 1]
    sns.countplot(data=flag_df, x="TARGET", order=order, ax=ax)
    total = len(flag_df)
    for patch in ax.patches:
        pct = patch.get_height() / total * 100
        ax.annotate(f"{pct:.2f}%", (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                    ha="center", va="bottom")
    ax.set_title("Target Distribution")
    plt.tight_layout()
    plt.show()


def default_rate_by_category(df: pd.DataFrame, category_col: str, target_col: str = "TARGET", top_n: int = 20) -> pd.DataFrame:
    temp = df[[category_col, target_col]].dropna()
    stats = (
        temp.groupby(category_col)[target_col]
        .agg(["count", "mean"])
        .rename(columns={"mean": "default_rate"})
        .sort_values(["default_rate", "count"], ascending=[False, False])
    )
    display(stats.head(top_n))
    plot_df = stats[stats["count"] >= max(20, int(0.001 * len(temp)))].head(top_n).reset_index()
    if len(plot_df):
        plt.figure(figsize=(9, max(4, 0.35 * len(plot_df))))
        sns.barplot(data=plot_df, y=category_col, x="default_rate")
        plt.title(f"Default Rate by {category_col}")
        plt.tight_layout()
        plt.show()
    return stats


def plot_numeric_target_views(df: pd.DataFrame, cols: list[str], target_col: str = "TARGET") -> None:
    for col in cols:
        if col not in df.columns:
            continue
        plot_df = df[[col, target_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if plot_df.empty:
            continue
        capped = plot_df[col].clip(plot_df[col].quantile(0.005), plot_df[col].quantile(0.995))
        plot_df = plot_df.assign(**{f"{col}_capped": capped, f"log1p_{col}": np.log1p(np.clip(capped, 0, None))})

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        sns.boxplot(data=plot_df, x=target_col, y=f"{col}_capped", ax=axes[0])
        axes[0].set_title(f"{col} by target (capped)")
        sns.boxplot(data=plot_df, x=target_col, y=f"log1p_{col}", ax=axes[1])
        axes[1].set_title(f"log1p({col}) by target")
        plt.tight_layout()
        plt.show()


def decile_target_table(df: pd.DataFrame, col: str, target_col: str = "TARGET", q: int = 10) -> pd.DataFrame | None:
    if col not in df.columns:
        return None
    temp = df[[col, target_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if temp[col].nunique() < 3:
        return None
    temp["bucket"] = pd.qcut(temp[col], q=q, duplicates="drop")
    out = temp.groupby("bucket", observed=True)[target_col].agg(["count", "mean"]).rename(columns={"mean": "default_rate"})
    print(f"\nDefault rate by {col} decile:")
    display(out)
    return out

# %%
# 08. EDA on the raw tables

plot_target_distribution(train_flag)

print("\nDefault rate by contract type:")
contract_rates = default_rate_by_category(train_flag, "NAME_CONTRACT_TYPE", top_n=10)

accounts_target = accounts_train.merge(train_flag[["uid", "TARGET"]], on="uid", how="left")
enquiries_target = enquiries_train.merge(train_flag[["uid", "TARGET"]], on="uid", how="left")

print("\nDefault rate by account credit_type:")
credit_type_rates = default_rate_by_category(accounts_target, "credit_type", top_n=15)

print("\nDefault rate by enquiry_type:")
enquiry_type_rates = default_rate_by_category(enquiries_target, "enquiry_type", top_n=20)

simple_account_eda = (
    accounts_train.groupby("uid")
    .agg(
        acct_count=("uid", "size"),
        acct_loan_sum=("loan_amount", "sum"),
        acct_loan_mean=("loan_amount", "mean"),
        acct_overdue_sum=("amount_overdue", "sum"),
        acct_overdue_max=("amount_overdue", "max"),
        acct_open_count=("closed_date", lambda s: s.isna().sum()),
    )
    .reset_index()
)

simple_enquiry_eda = (
    enquiries_train.groupby("uid")
    .agg(
        enq_count=("uid", "size"),
        enq_amt_sum=("enquiry_amt", "sum"),
        enq_amt_mean=("enquiry_amt", "mean"),
        enq_amt_max=("enquiry_amt", "max"),
    )
    .reset_index()
)

eda_user = (
    train_flag[["uid", "TARGET", "NAME_CONTRACT_TYPE"]]
    .merge(simple_account_eda, on="uid", how="left")
    .merge(simple_enquiry_eda, on="uid", how="left")
)
eda_user["has_accounts"] = eda_user["acct_count"].notna().astype(int)
eda_user = eda_user.fillna(0)
eda_user["acct_open_ratio"] = eda_user["acct_open_count"] / (eda_user["acct_count"] + 1)
eda_user["enq_per_account"] = eda_user["enq_count"] / (eda_user["acct_count"] + 1)
eda_user["overdue_to_loan"] = eda_user["acct_overdue_sum"] / (eda_user["acct_loan_sum"] + 1)

print("\nBorrower-level aggregates used for quick EDA:")
display(eda_user.head())

plot_numeric_target_views(
    eda_user,
    [
        "acct_count",
        "acct_loan_sum",
        "acct_overdue_sum",
        "acct_overdue_max",
        "enq_count",
        "enq_amt_sum",
        "enq_per_account",
        "overdue_to_loan",
    ],
)

for col in ["acct_count", "acct_loan_sum", "acct_overdue_sum", "enq_count", "enq_amt_sum", "enq_per_account"]:
    decile_target_table(eda_user, col)

plt.figure(figsize=(8, 5))
nonzero_overdue = accounts_train.loc[accounts_train["amount_overdue"] > 0, "amount_overdue"]
sns.histplot(np.log1p(nonzero_overdue), bins=60)
plt.title("Non-zero Overdue Distribution - log1p")
plt.tight_layout()
plt.show()

# %%
# 09. Payment history / DPD checks


def empty_dpd_features() -> dict[str, float]:
    out = {
        "hist_months": 0,
        "dpd_latest": 0,
        "dpd_max": 0,
        "dpd_mean": 0.0,
        "dpd_std": 0.0,
        "dpd_sum": 0.0,
        "dpd_months_pos": 0,
        "dpd_months_30p": 0,
        "dpd_months_60p": 0,
        "dpd_months_90p": 0,
        "dpd_months_180p": 0,
        "dpd_any_pos": 0,
        "dpd_any_30p": 0,
        "dpd_any_60p": 0,
        "dpd_any_90p": 0,
        "dpd_any_180p": 0,
        "dpd_pos_ratio": 0.0,
        "dpd_30p_ratio": 0.0,
        "dpd_90p_ratio": 0.0,
        "current_bad_streak": 0,
        "max_bad_streak": 0,
        "max_good_streak": 0,
        "months_since_last_dpd": 999,
        "dpd_recent6_minus_prev6": 0.0,
        "dpd_recent12_minus_prev12": 0.0,
        "dpd_recent_weighted_mean": 0.0,
    }
    for window in DPD_WINDOWS:
        out[f"recent_{window}m_max"] = 0
        out[f"recent_{window}m_mean"] = 0.0
        out[f"recent_{window}m_sum"] = 0.0
        out[f"recent_{window}m_pos_months"] = 0
        out[f"recent_{window}m_30p_months"] = 0
        out[f"recent_{window}m_90p_months"] = 0
    return out


DPD_EMPTY = empty_dpd_features()


def longest_true_streak(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def dpd_features_from_string(hist: object) -> dict[str, float]:
    if pd.isna(hist):
        return DPD_EMPTY.copy()
    s = str(hist).strip()
    if not s or s.lower() in {"nan", "none", "nat"}:
        return DPD_EMPTY.copy()
    if not s.isdigit():
        s = re.sub(r"\D", "", s)
    if not s or len(s) % 3 != 0:
        return DPD_EMPTY.copy()

    vals = np.array([int(s[i:i + 3]) for i in range(0, len(s), 3)], dtype=np.float32)
    n = len(vals)
    if n == 0:
        return DPD_EMPTY.copy()

    pos = vals > 0
    ge30 = vals >= 30
    ge60 = vals >= 60
    ge90 = vals >= 90
    ge180 = vals >= 180
    positive_indices = np.where(pos)[0]
    weights = np.linspace(1.0, 2.0, n)

    out = {
        "hist_months": int(n),
        "dpd_latest": float(vals[-1]),
        "dpd_max": float(vals.max()),
        "dpd_mean": float(vals.mean()),
        "dpd_std": float(vals.std()),
        "dpd_sum": float(vals.sum()),
        "dpd_months_pos": int(pos.sum()),
        "dpd_months_30p": int(ge30.sum()),
        "dpd_months_60p": int(ge60.sum()),
        "dpd_months_90p": int(ge90.sum()),
        "dpd_months_180p": int(ge180.sum()),
        "dpd_any_pos": int(pos.any()),
        "dpd_any_30p": int(ge30.any()),
        "dpd_any_60p": int(ge60.any()),
        "dpd_any_90p": int(ge90.any()),
        "dpd_any_180p": int(ge180.any()),
        "dpd_pos_ratio": float(pos.mean()),
        "dpd_30p_ratio": float(ge30.mean()),
        "dpd_90p_ratio": float(ge90.mean()),
        "current_bad_streak": int(next((i for i, value in enumerate(pos[::-1]) if not value), n)),
        "max_bad_streak": longest_true_streak(pos),
        "max_good_streak": longest_true_streak(~pos),
        "months_since_last_dpd": int(n - 1 - positive_indices[-1]) if len(positive_indices) else 999,
        "dpd_recent_weighted_mean": float(np.average(vals, weights=weights)),
    }

    recent6 = vals[-6:] if n >= 6 else vals
    prev6 = vals[-12:-6] if n >= 12 else vals[: max(0, n - len(recent6))]
    recent12 = vals[-12:] if n >= 12 else vals
    prev12 = vals[-24:-12] if n >= 24 else vals[: max(0, n - len(recent12))]
    out["dpd_recent6_minus_prev6"] = float(recent6.mean() - prev6.mean()) if len(prev6) else 0.0
    out["dpd_recent12_minus_prev12"] = float(recent12.mean() - prev12.mean()) if len(prev12) else 0.0

    for window in DPD_WINDOWS:
        tail = vals[-window:] if n >= window else vals
        out[f"recent_{window}m_max"] = float(tail.max()) if len(tail) else 0
        out[f"recent_{window}m_mean"] = float(tail.mean()) if len(tail) else 0.0
        out[f"recent_{window}m_sum"] = float(tail.sum()) if len(tail) else 0.0
        out[f"recent_{window}m_pos_months"] = int((tail > 0).sum()) if len(tail) else 0
        out[f"recent_{window}m_30p_months"] = int((tail >= 30).sum()) if len(tail) else 0
        out[f"recent_{window}m_90p_months"] = int((tail >= 90).sum()) if len(tail) else 0

    return out


dpd_sample = accounts_train[["uid", "payment_hist_string"]].sample(min(200000, len(accounts_train)), random_state=RANDOM_STATE)
dpd_sample_features = pd.DataFrame.from_records(dpd_sample["payment_hist_string"].map(dpd_features_from_string))
dpd_sample_eda = pd.concat([dpd_sample[["uid"]].reset_index(drop=True), dpd_sample_features], axis=1)
dpd_sample_eda = dpd_sample_eda.groupby("uid").agg({"dpd_max": "max", "dpd_months_pos": "sum", "recent_12m_max": "max"}).reset_index()
dpd_sample_eda = dpd_sample_eda.merge(train_flag[["uid", "TARGET"]], on="uid", how="left")

for col in ["dpd_max", "dpd_months_pos", "recent_12m_max"]:
    decile_target_table(dpd_sample_eda, col)

plot_numeric_target_views(dpd_sample_eda, ["dpd_max", "dpd_months_pos", "recent_12m_max"])

# %%
# 10. Feature helper functions


def clean_token(value: object) -> str:
    token = re.sub(r"[^0-9a-zA-Z]+", "_", str(value)).strip("_").lower()
    return token or "missing"


def make_unique_columns(columns: list[object]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for col in columns:
        base = clean_token(col)
        count = seen.get(base, 0)
        out.append(base if count == 0 else f"{base}_{count}")
        seen[base] = count + 1
    return out


def safe_divide(num: pd.Series, den: pd.Series, fill: float = 0.0) -> pd.Series:
    result = num / den.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill)


def get_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def category_mix_features(
    df: pd.DataFrame,
    uid_col: str,
    cat_col: str,
    amount_col: str,
    top_values: list[str],
    prefix: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[uid_col])

    categories = list(top_values) + ["OTHER"]
    top_set = set(top_values)
    work = df[[uid_col, cat_col, amount_col]].copy()
    work[f"{cat_col}_top"] = np.where(work[cat_col].isin(top_set), work[cat_col], "OTHER")

    counts = pd.crosstab(work[uid_col], work[f"{cat_col}_top"])
    for category in categories:
        if category not in counts.columns:
            counts[category] = 0
    counts = counts[categories].astype("float32")

    total = counts.sum(axis=1).replace(0, np.nan)
    shares = counts.div(total, axis=0).fillna(0)
    probs = shares.replace(0, np.nan)
    entropy = (-(probs * np.log(probs)).sum(axis=1) / np.log(max(len(categories), 2))).fillna(0)

    amounts = work.pivot_table(index=uid_col, columns=f"{cat_col}_top", values=amount_col, aggfunc="sum", fill_value=0)
    for category in categories:
        if category not in amounts.columns:
            amounts[category] = 0
    amounts = amounts[categories].astype("float32")

    counts.columns = [f"{prefix}_{clean_token(category)}_count" for category in categories]
    shares.columns = [f"{prefix}_{clean_token(category)}_share" for category in categories]
    amounts.columns = [f"{prefix}_{clean_token(category)}_amount_sum" for category in categories]

    out = pd.concat([counts, shares, amounts], axis=1)
    out[f"{prefix}_entropy"] = entropy.astype("float32")
    out[f"{prefix}_unique_count"] = (counts > 0).sum(axis=1).astype("float32")
    out = out.reset_index()
    return out


def flatten_multiindex_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [f"{prefix}_{clean_token(a)}_{clean_token(b)}" for a, b in df.columns]
    return df

# %%
# 11. Account features


def add_account_row_features(df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    df["is_open"] = df["closed_date"].isna().astype(np.int8)
    df["is_closed"] = 1 - df["is_open"]
    df["has_overdue"] = (df["amount_overdue"] > 0).astype(np.int8)
    df["zero_loan_flag"] = (df["loan_amount"] <= 0).astype(np.int8)
    df["open_loan_amount"] = df["loan_amount"] * df["is_open"]
    df["closed_loan_amount"] = df["loan_amount"] * df["is_closed"]
    df["overdue_to_loan"] = safe_divide(df["amount_overdue"], df["loan_amount"])

    df["days_since_open"] = (reference_date - df["open_date"]).dt.days.clip(lower=0)
    df["days_since_close"] = (reference_date - df["closed_date"]).dt.days.clip(lower=0)
    df["account_end_date"] = df["closed_date"].fillna(reference_date)
    df["account_duration_days"] = (df["account_end_date"] - df["open_date"]).dt.days.clip(lower=0)

    for window in RECENCY_WINDOWS:
        df[f"opened_last_{window}d"] = ((df["days_since_open"] <= window) & df["open_date"].notna()).astype(np.int8)
        df[f"closed_last_{window}d"] = ((df["days_since_close"] <= window) & df["closed_date"].notna()).astype(np.int8)

    df["loan_amount_log1p"] = np.log1p(df["loan_amount"])
    df["amount_overdue_log1p"] = np.log1p(df["amount_overdue"])
    return df


def aggregate_dpd_features(accounts_df: pd.DataFrame) -> pd.DataFrame:
    hist = accounts_df["payment_hist_string"].fillna("").astype(str)
    unique_histories = hist.unique()
    print(f"Parsing {len(unique_histories):,} unique payment history strings")
    dpd_cache = {value: dpd_features_from_string(value) for value in unique_histories}
    dpd_rows = pd.DataFrame.from_records(hist.map(dpd_cache), index=accounts_df.index)
    dpd_rows = pd.concat([accounts_df[["uid"]].reset_index(drop=True), dpd_rows.reset_index(drop=True)], axis=1)

    dpd_cols = [col for col in dpd_rows.columns if col != "uid"]
    dpd_agg = dpd_rows.groupby("uid")[dpd_cols].agg(["sum", "mean", "max"])
    dpd_agg = flatten_multiindex_columns(dpd_agg, "acct_dpd").reset_index()
    return dpd_agg


def build_account_features(accounts_df: pd.DataFrame, top_credit_types: list[str], reference_date: pd.Timestamp) -> pd.DataFrame:
    df = add_account_row_features(accounts_df, reference_date)
    group = df.groupby("uid", sort=False)

    basic = group.agg(
        acct_count=("uid", "size"),
        acct_open_count=("is_open", "sum"),
        acct_closed_count=("is_closed", "sum"),
        acct_has_overdue_count=("has_overdue", "sum"),
        acct_zero_loan_count=("zero_loan_flag", "sum"),
        acct_invalid_close_count=("closed_date_before_open_flag", "sum"),
        acct_loan_missing_count=("loan_amount_missing", "sum"),
        acct_overdue_missing_count=("amount_overdue_missing", "sum"),
        acct_loan_sum=("loan_amount", "sum"),
        acct_loan_mean=("loan_amount", "mean"),
        acct_loan_median=("loan_amount", "median"),
        acct_loan_max=("loan_amount", "max"),
        acct_loan_min=("loan_amount", "min"),
        acct_loan_std=("loan_amount", "std"),
        acct_open_loan_sum=("open_loan_amount", "sum"),
        acct_closed_loan_sum=("closed_loan_amount", "sum"),
        acct_overdue_sum=("amount_overdue", "sum"),
        acct_overdue_mean=("amount_overdue", "mean"),
        acct_overdue_max=("amount_overdue", "max"),
        acct_overdue_std=("amount_overdue", "std"),
        acct_overdue_to_loan_mean=("overdue_to_loan", "mean"),
        acct_overdue_to_loan_max=("overdue_to_loan", "max"),
        acct_days_since_open_min=("days_since_open", "min"),
        acct_days_since_open_mean=("days_since_open", "mean"),
        acct_days_since_open_max=("days_since_open", "max"),
        acct_days_since_close_min=("days_since_close", "min"),
        acct_days_since_close_mean=("days_since_close", "mean"),
        acct_duration_min=("account_duration_days", "min"),
        acct_duration_mean=("account_duration_days", "mean"),
        acct_duration_max=("account_duration_days", "max"),
        acct_duration_std=("account_duration_days", "std"),
    ).reset_index()

    basic["acct_open_ratio"] = safe_divide(basic["acct_open_count"], basic["acct_count"])
    basic["acct_overdue_account_ratio"] = safe_divide(basic["acct_has_overdue_count"], basic["acct_count"])
    basic["acct_zero_loan_ratio"] = safe_divide(basic["acct_zero_loan_count"], basic["acct_count"])
    basic["acct_open_loan_ratio"] = safe_divide(basic["acct_open_loan_sum"], basic["acct_loan_sum"])
    basic["acct_overdue_to_total_loan"] = safe_divide(basic["acct_overdue_sum"], basic["acct_loan_sum"])

    window_cols = [f"opened_last_{w}d" for w in RECENCY_WINDOWS] + [f"closed_last_{w}d" for w in RECENCY_WINDOWS]
    windows = group[window_cols].sum().add_prefix("acct_").reset_index()

    dpd = aggregate_dpd_features(df)
    mix = category_mix_features(df, "uid", "credit_type", "loan_amount", top_credit_types, "acct_credit_type")

    features = (
        basic.merge(windows, on="uid", how="left")
        .merge(dpd, on="uid", how="left")
        .merge(mix, on="uid", how="left")
    )
    return features

# %%
# 12. Enquiry features


def add_enquiry_row_features(df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    df["days_since_enquiry"] = (reference_date - df["enquiry_date"]).dt.days.clip(lower=0)
    df["enquiry_amt_log1p"] = np.log1p(df["enquiry_amt"])
    for window in RECENCY_WINDOWS:
        df[f"enquiry_last_{window}d"] = ((df["days_since_enquiry"] <= window) & df["enquiry_date"].notna()).astype(np.int8)
        df[f"enquiry_amt_last_{window}d"] = df["enquiry_amt"] * df[f"enquiry_last_{window}d"]
    return df


def build_enquiry_features(enquiries_df: pd.DataFrame, top_enquiry_types: list[str], reference_date: pd.Timestamp) -> pd.DataFrame:
    df = add_enquiry_row_features(enquiries_df, reference_date)
    group = df.groupby("uid", sort=False)

    basic = group.agg(
        enq_count=("uid", "size"),
        enq_amt_sum=("enquiry_amt", "sum"),
        enq_amt_mean=("enquiry_amt", "mean"),
        enq_amt_median=("enquiry_amt", "median"),
        enq_amt_max=("enquiry_amt", "max"),
        enq_amt_min=("enquiry_amt", "min"),
        enq_amt_std=("enquiry_amt", "std"),
        enq_amt_missing_count=("enquiry_amt_missing", "sum"),
        enq_days_since_min=("days_since_enquiry", "min"),
        enq_days_since_mean=("days_since_enquiry", "mean"),
        enq_days_since_max=("days_since_enquiry", "max"),
        enq_days_since_std=("days_since_enquiry", "std"),
    ).reset_index()

    window_cols = [f"enquiry_last_{w}d" for w in RECENCY_WINDOWS] + [f"enquiry_amt_last_{w}d" for w in RECENCY_WINDOWS]
    windows = group[window_cols].sum().add_prefix("enq_").reset_index()

    mix = category_mix_features(df, "uid", "enquiry_type", "enquiry_amt", top_enquiry_types, "enq_type")

    features = basic.merge(windows, on="uid", how="left").merge(mix, on="uid", how="left")
    for window in RECENCY_WINDOWS:
        features[f"enq_amt_avg_last_{window}d"] = safe_divide(
            features[f"enq_enquiry_amt_last_{window}d"],
            features[f"enq_enquiry_last_{window}d"],
        )
    return features

# %%
# 13. Build train and test feature tables

TOP_CREDIT_TYPES = accounts_train["credit_type"].value_counts().head(TOP_N_CREDIT_TYPES).index.tolist()
TOP_ENQUIRY_TYPES = enquiries_train["enquiry_type"].value_counts().head(TOP_N_ENQUIRY_TYPES).index.tolist()

print("Top credit types:", TOP_CREDIT_TYPES)
print("Top enquiry types:", TOP_ENQUIRY_TYPES)

account_features_train = build_account_features(accounts_train, TOP_CREDIT_TYPES, REFERENCE_DATE)
account_features_test = build_account_features(accounts_test, TOP_CREDIT_TYPES, REFERENCE_DATE)
enquiry_features_train = build_enquiry_features(enquiries_train, TOP_ENQUIRY_TYPES, REFERENCE_DATE)
enquiry_features_test = build_enquiry_features(enquiries_test, TOP_ENQUIRY_TYPES, REFERENCE_DATE)

features_train = account_features_train.merge(enquiry_features_train, on="uid", how="outer")
features_test = account_features_test.merge(enquiry_features_test, on="uid", how="outer")

print("Account features train:", account_features_train.shape)
print("Enquiry features train:", enquiry_features_train.shape)
print("Combined features train:", features_train.shape)
print("Combined features test :", features_test.shape)

# %%
# 14. Cross features after joining flags


def add_cross_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()

    df["has_prior_accounts"] = get_series(df, "acct_count").notna().astype(np.int8)
    df["has_prior_enquiries"] = get_series(df, "enq_count").notna().astype(np.int8)

    acct_count = get_series(df, "acct_count").fillna(0)
    enq_count = get_series(df, "enq_count").fillna(0)
    acct_loan_sum = get_series(df, "acct_loan_sum").fillna(0)
    acct_open_loan_sum = get_series(df, "acct_open_loan_sum").fillna(0)
    acct_overdue_sum = get_series(df, "acct_overdue_sum").fillna(0)
    enq_amt_sum = get_series(df, "enq_amt_sum").fillna(0)

    df["ratio_enquiries_per_account"] = enq_count / (acct_count + 1)
    df["ratio_enq_amt_to_loan_amt"] = enq_amt_sum / (acct_loan_sum + 1)
    df["ratio_overdue_to_loan_amt"] = acct_overdue_sum / (acct_loan_sum + 1)
    df["ratio_open_loan_to_total_loan"] = acct_open_loan_sum / (acct_loan_sum + 1)
    df["ratio_overdue_per_account"] = acct_overdue_sum / (acct_count + 1)
    df["ratio_enq_amt_per_account"] = enq_amt_sum / (acct_count + 1)

    hist_months = get_series(df, "acct_dpd_hist_months_sum").fillna(0)
    dpd_pos = get_series(df, "acct_dpd_dpd_months_pos_sum").fillna(0)
    dpd_30p = get_series(df, "acct_dpd_dpd_months_30p_sum").fillna(0)
    dpd_90p = get_series(df, "acct_dpd_dpd_months_90p_sum").fillna(0)
    recent12_max = get_series(df, "acct_dpd_recent_12m_max_max").fillna(0)
    dpd_max = get_series(df, "acct_dpd_dpd_max_max").fillna(0)

    df["ratio_dpd_pos_months"] = dpd_pos / (hist_months + 1)
    df["ratio_dpd_30p_months"] = dpd_30p / (hist_months + 1)
    df["ratio_dpd_90p_months"] = dpd_90p / (hist_months + 1)
    df["risk_recent12_dpd_x_overdue_log"] = recent12_max * np.log1p(acct_overdue_sum)
    df["risk_max_dpd_x_open_loan_log"] = dpd_max * np.log1p(acct_open_loan_sum)
    df["risk_enq_pressure_x_dpd"] = df["ratio_enquiries_per_account"] * np.log1p(dpd_max)
    df["risk_enq_amount_pressure_x_overdue"] = df["ratio_enq_amt_to_loan_amt"] * np.log1p(acct_overdue_sum)

    # Most amount/count columns are right-skewed, so I keep both raw and log versions.
    skip = {"TARGET"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    log_candidates = [
        col for col in numeric_cols
        if col not in skip
        and any(token in col for token in ["sum", "mean", "max", "amount", "loan", "overdue", "count", "days_since", "duration"])
    ]
    for col in log_candidates:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.min(skipna=True) >= 0:
            df[f"log1p_{col}"] = np.log1p(values.fillna(0).clip(lower=0))

    return df


train_model = train_flag.merge(features_train, on="uid", how="left")
test_model = test_flag.merge(features_test, on="uid", how="left")

train_model = add_cross_features(train_model)
test_model = add_cross_features(test_model)

print("Train model frame:", train_model.shape)
print("Test model frame :", test_model.shape)
display(train_model.head())

# %%
# 15. Feature checks before modelling

feature_missing = (
    train_model.drop(columns=["uid", "TARGET"])
    .isna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .head(40)
)
print("Top missing feature percentages:")
display(feature_missing)

numeric_corr_df = train_model.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
target_corr = numeric_corr_df.corr(numeric_only=True)["TARGET"].drop("TARGET").sort_values()

print("\nMost negative correlations with TARGET:")
display(target_corr.head(25))
print("\nMost positive correlations with TARGET:")
display(target_corr.tail(25).sort_values(ascending=False))

important_eda_cols = [
    "ratio_dpd_pos_months",
    "ratio_dpd_30p_months",
    "ratio_dpd_90p_months",
    "ratio_enquiries_per_account",
    "ratio_enq_amt_to_loan_amt",
    "ratio_overdue_to_loan_amt",
    "risk_recent12_dpd_x_overdue_log",
    "risk_enq_pressure_x_dpd",
]
for col in important_eda_cols:
    decile_target_table(train_model, col)

# %%
# 16. Preprocessing and train/test alignment


def encode_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = "TARGET"):
    raw_feature_cols = [col for col in train_df.columns if col not in ["uid", target_col]]
    X_raw = train_df[raw_feature_cols].copy()
    X_test_raw = test_df.reindex(columns=raw_feature_cols).copy()
    y = train_df[target_col].astype(int).values

    cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    print("Categorical columns:", cat_cols)

    X_enc = pd.get_dummies(X_raw, columns=cat_cols, dummy_na=True)
    X_test_enc = pd.get_dummies(X_test_raw, columns=cat_cols, dummy_na=True)
    X_test_enc = X_test_enc.reindex(columns=X_enc.columns, fill_value=0)

    clean_cols = make_unique_columns(list(X_enc.columns))
    X_enc.columns = clean_cols
    X_test_enc.columns = clean_cols

    X_enc = X_enc.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_test_enc = X_test_enc.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return X_enc, X_test_enc, y


def fit_transform_numeric(X_train: pd.DataFrame, X_test: pd.DataFrame):
    lower_q, upper_q = CLIP_QUANTILES
    medians = X_train.median(axis=0).fillna(0)
    lower = X_train.quantile(lower_q).fillna(medians)
    upper = X_train.quantile(upper_q).fillna(medians)

    X_train_clean = X_train.fillna(medians).clip(lower=lower, upper=upper, axis=1)
    X_test_clean = X_test.fillna(medians).clip(lower=lower, upper=upper, axis=1)

    constant_cols = X_train_clean.columns[X_train_clean.nunique(dropna=False) <= 1].tolist()
    if constant_cols:
        print(f"Dropping {len(constant_cols)} constant columns")
        X_train_clean = X_train_clean.drop(columns=constant_cols)
        X_test_clean = X_test_clean.drop(columns=constant_cols)

    return X_train_clean.astype("float32"), X_test_clean.astype("float32"), {
        "medians": medians,
        "lower": lower,
        "upper": upper,
        "constant_cols": constant_cols,
    }


X_encoded, X_test_encoded, y = encode_train_test(train_model, test_model)
X, X_test, preprocess_artifacts = fit_transform_numeric(X_encoded, X_test_encoded)

print("Prepared X     :", X.shape)
print("Prepared X_test:", X_test.shape)
print("Target bad rate:", y.mean().round(5))

assert X.columns.equals(X_test.columns)
assert np.isfinite(X.to_numpy()).all()
assert np.isfinite(X_test.to_numpy()).all()

# %%
# 17. Cross-validation helpers

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def auc_report(name: str, y_true: np.ndarray, oof_pred: np.ndarray, fold_scores: list[float]) -> float:
    auc = roc_auc_score(y_true, oof_pred)
    print(f"\n{name} OOF AUC: {auc:.6f}")
    print(f"{name} fold AUCs:", [round(score, 6) for score in fold_scores])
    print(f"{name} fold mean/std: {np.mean(fold_scores):.6f} +/- {np.std(fold_scores):.6f}")
    return auc


def run_lr_cv(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_holdout: pd.DataFrame | None,
    params: dict,
    name: str,
) -> tuple[np.ndarray, np.ndarray | None, list, list[float]]:
    oof = np.zeros(len(y_train), dtype="float64")
    test_pred = np.zeros(len(X_holdout), dtype="float64") if X_holdout is not None else None
    models = []
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
        model_params = {
            "class_weight": "balanced",
            "random_state": RANDOM_STATE + fold,
            "max_iter": 5000,
        }
        model_params.update(params)
        if model_params.get("solver") in {"saga", "liblinear"}:
            model_params["n_jobs"] = -1 if model_params.get("solver") == "saga" else None

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**model_params)),
        ])
        model.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        valid_pred = model.predict_proba(X_train.iloc[va_idx])[:, 1]
        oof[va_idx] = valid_pred
        score = roc_auc_score(y_train[va_idx], valid_pred)
        fold_scores.append(score)
        models.append(model)
        if X_holdout is not None:
            test_pred += model.predict_proba(X_holdout)[:, 1] / N_FOLDS
        print(f"{name} fold {fold}: {score:.6f}")

    return oof, test_pred, models, fold_scores


def run_lgbm_cv(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_holdout: pd.DataFrame | None,
    params: dict,
    name: str,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, list, list[float], np.ndarray]:
    oof = np.zeros(len(y_train), dtype="float64")
    test_pred = np.zeros(len(X_holdout), dtype="float64") if X_holdout is not None else None
    models = []
    fold_scores = []
    importance = np.zeros(X_train.shape[1], dtype="float64")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
        base_params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "n_estimators": 5000,
            "learning_rate": 0.03,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE + fold,
            "n_jobs": -1,
            "verbose": -1,
            "importance_type": "gain",
            "force_col_wise": True,
        }
        model_params = {**base_params, **params}
        model = lgb.LGBMClassifier(**model_params)
        model.fit(
            X_train.iloc[tr_idx],
            y_train[tr_idx],
            eval_set=[(X_train.iloc[va_idx], y_train[va_idx])],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        valid_pred = model.predict_proba(X_train.iloc[va_idx])[:, 1]
        oof[va_idx] = valid_pred
        score = roc_auc_score(y_train[va_idx], valid_pred)
        fold_scores.append(score)
        models.append(model)
        importance += model.feature_importances_ / N_FOLDS
        if X_holdout is not None:
            test_pred += model.predict_proba(X_holdout)[:, 1] / N_FOLDS
        if verbose:
            print(f"{name} fold {fold}: {score:.6f} | best_iter={model.best_iteration_}")

    return oof, test_pred, models, fold_scores, importance


def run_xgb_cv(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_holdout: pd.DataFrame | None,
    params: dict,
    name: str,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, list, list[float], np.ndarray]:
    oof = np.zeros(len(y_train), dtype="float64")
    test_pred = np.zeros(len(X_holdout), dtype="float64") if X_holdout is not None else None
    models = []
    fold_scores = []
    importance = np.zeros(X_train.shape[1], dtype="float64")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
        y_tr = y_train[tr_idx]
        scale_pos_weight = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1.0)
        base_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "n_estimators": 5000,
            "learning_rate": 0.03,
            "scale_pos_weight": scale_pos_weight,
            "random_state": RANDOM_STATE + fold,
            "n_jobs": -1,
            "verbosity": 0,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        }
        model_params = {**base_params, **params}
        model = xgb.XGBClassifier(**model_params)
        model.fit(
            X_train.iloc[tr_idx],
            y_train[tr_idx],
            eval_set=[(X_train.iloc[va_idx], y_train[va_idx])],
            verbose=False,
        )
        valid_pred = model.predict_proba(X_train.iloc[va_idx])[:, 1]
        oof[va_idx] = valid_pred
        score = roc_auc_score(y_train[va_idx], valid_pred)
        fold_scores.append(score)
        models.append(model)
        importance += model.feature_importances_ / N_FOLDS
        if X_holdout is not None:
            test_pred += model.predict_proba(X_holdout)[:, 1] / N_FOLDS
        if verbose:
            best_iter = getattr(model, "best_iteration", None)
            print(f"{name} fold {fold}: {score:.6f} | best_iter={best_iter}")

    return oof, test_pred, models, fold_scores, importance

# %%
# 18. Logistic Regression

print("Logistic Regression baseline")
lr_baseline_params = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
}
oof_lr_base, pred_lr_base, lr_base_models, lr_base_scores = run_lr_cv(
    X, y, X_test, lr_baseline_params, "LR baseline"
)
auc_lr_base = auc_report("LR baseline", y, oof_lr_base, lr_base_scores)

print("\nTrying a small LR parameter search")
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000, n_jobs=-1)),
])

lr_param_dist = {
    "clf__C": loguniform(1e-4, 50),
    "clf__penalty": ["l1", "l2", "elasticnet"],
    "clf__solver": ["saga"],
    "clf__l1_ratio": uniform(0.0, 1.0),
}

lr_search = RandomizedSearchCV(
    estimator=lr_pipe,
    param_distributions=lr_param_dist,
    n_iter=CFG["lr_iter"],
    scoring="roc_auc",
    cv=3,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
    refit=False,
)
lr_search.fit(X, y)

best_lr_params = {key.replace("clf__", ""): value for key, value in lr_search.best_params_.items()}
print("LR params used:", best_lr_params)
print("LR search CV AUC:", round(lr_search.best_score_, 6))

oof_lr, pred_lr, lr_models, lr_scores = run_lr_cv(X, y, X_test, best_lr_params, "LR tuned")
auc_lr = auc_report("LR tuned", y, oof_lr, lr_scores)

# %%
# 19. LightGBM

lgb_param_dist = {
    "n_estimators": [2500, 3500, 5000],
    "learning_rate": loguniform(0.008, 0.08),
    "num_leaves": randint(24, 180),
    "max_depth": [-1, 3, 4, 5, 6, 7, 8, 10],
    "min_child_samples": randint(30, 260),
    "subsample": uniform(0.55, 0.45),
    "subsample_freq": randint(1, 8),
    "colsample_bytree": uniform(0.55, 0.45),
    "reg_alpha": loguniform(1e-4, 10),
    "reg_lambda": loguniform(1e-4, 20),
    "min_split_gain": uniform(0.0, 0.25),
}


def tune_lgbm_params() -> tuple[dict, pd.DataFrame]:
    results = []
    best_score = -np.inf
    best_params = None
    sampler = list(ParameterSampler(lgb_param_dist, n_iter=CFG["lgb_iter"], random_state=RANDOM_STATE))
    for i, params in enumerate(sampler, 1):
        print(f"\nLightGBM trial {i}/{len(sampler)}")
        oof_tmp, _, _, scores_tmp, _ = run_lgbm_cv(X, y, None, params, f"LGB tune {i}", verbose=False)
        score = roc_auc_score(y, oof_tmp)
        results.append({"iteration": i, "auc": score, **params})
        print(f"LGB trial {i} OOF AUC: {score:.6f}")
        if score > best_score:
            best_score = score
            best_params = params
    return best_params, pd.DataFrame(results).sort_values("auc", ascending=False)


best_lgb_params, lgb_tuning_results = tune_lgbm_params()
print("\nLightGBM params used:")
print(best_lgb_params)
display(lgb_tuning_results.head(10))

oof_lgb, pred_lgb, lgb_models, lgb_scores, lgb_importance = run_lgbm_cv(
    X, y, X_test, best_lgb_params, "LGB final"
)
auc_lgb = auc_report("LGB final", y, oof_lgb, lgb_scores)

# %%
# 20. XGBoost

xgb_param_dist = {
    "n_estimators": [2500, 3500, 5000],
    "learning_rate": loguniform(0.008, 0.08),
    "max_depth": randint(2, 8),
    "min_child_weight": loguniform(0.5, 30),
    "subsample": uniform(0.55, 0.45),
    "colsample_bytree": uniform(0.55, 0.45),
    "gamma": uniform(0.0, 5.0),
    "reg_alpha": loguniform(1e-4, 10),
    "reg_lambda": loguniform(1e-3, 30),
}


def tune_xgb_params() -> tuple[dict, pd.DataFrame]:
    results = []
    best_score = -np.inf
    best_params = None
    sampler = list(ParameterSampler(xgb_param_dist, n_iter=CFG["xgb_iter"], random_state=RANDOM_STATE + 7))
    for i, params in enumerate(sampler, 1):
        print(f"\nXGBoost trial {i}/{len(sampler)}")
        oof_tmp, _, _, scores_tmp, _ = run_xgb_cv(X, y, None, params, f"XGB tune {i}", verbose=False)
        score = roc_auc_score(y, oof_tmp)
        results.append({"iteration": i, "auc": score, **params})
        print(f"XGB trial {i} OOF AUC: {score:.6f}")
        if score > best_score:
            best_score = score
            best_params = params
    return best_params, pd.DataFrame(results).sort_values("auc", ascending=False)


best_xgb_params, xgb_tuning_results = tune_xgb_params()
print("\nXGBoost params used:")
print(best_xgb_params)
display(xgb_tuning_results.head(10))

oof_xgb, pred_xgb, xgb_models, xgb_scores, xgb_importance = run_xgb_cv(
    X, y, X_test, best_xgb_params, "XGB final"
)
auc_xgb = auc_report("XGB final", y, oof_xgb, xgb_scores)

# %%
# 21. OOF blending

model_scores = {
    "lr_base": auc_lr_base,
    "lr": auc_lr,
    "lgb": auc_lgb,
    "xgb": auc_xgb,
}
score_table = pd.DataFrame({"model": list(model_scores.keys()), "oof_auc": list(model_scores.values())}).sort_values("oof_auc", ascending=False)
display(score_table)

oof_map = {
    "lr": oof_lr,
    "lgb": oof_lgb,
    "xgb": oof_xgb,
}
test_map = {
    "lr": pred_lr,
    "lgb": pred_lgb,
    "xgb": pred_xgb,
}


def rank_normalize(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).values


blend_oof = {}
blend_test = {}

blend_oof["simple_mean"] = np.mean(np.column_stack([oof_lr, oof_lgb, oof_xgb]), axis=1)
blend_test["simple_mean"] = np.mean(np.column_stack([pred_lr, pred_lgb, pred_xgb]), axis=1)

tree_weight_total = auc_lgb + auc_xgb
blend_oof["tree_weighted"] = (auc_lgb / tree_weight_total) * oof_lgb + (auc_xgb / tree_weight_total) * oof_xgb
blend_test["tree_weighted"] = (auc_lgb / tree_weight_total) * pred_lgb + (auc_xgb / tree_weight_total) * pred_xgb

all_weight_total = auc_lr + auc_lgb + auc_xgb
blend_oof["auc_weighted_all"] = (
    (auc_lr / all_weight_total) * oof_lr
    + (auc_lgb / all_weight_total) * oof_lgb
    + (auc_xgb / all_weight_total) * oof_xgb
)
blend_test["auc_weighted_all"] = (
    (auc_lr / all_weight_total) * pred_lr
    + (auc_lgb / all_weight_total) * pred_lgb
    + (auc_xgb / all_weight_total) * pred_xgb
)

blend_oof["rank_mean"] = np.mean(np.column_stack([rank_normalize(oof_lr), rank_normalize(oof_lgb), rank_normalize(oof_xgb)]), axis=1)
blend_test["rank_mean"] = np.mean(np.column_stack([rank_normalize(pred_lr), rank_normalize(pred_lgb), rank_normalize(pred_xgb)]), axis=1)

# Meta model is trained only on OOF predictions, so the blend score stays honest.
meta_X = np.column_stack([oof_lr, oof_lgb, oof_xgb])
meta_test_X = np.column_stack([pred_lr, pred_lgb, pred_xgb])
meta_oof = np.zeros(len(y), dtype="float64")
for fold, (tr_idx, va_idx) in enumerate(skf.split(meta_X, y), 1):
    meta = LogisticRegression(C=1.0, solver="lbfgs", class_weight="balanced", max_iter=1000)
    meta.fit(meta_X[tr_idx], y[tr_idx])
    meta_oof[va_idx] = meta.predict_proba(meta_X[va_idx])[:, 1]
meta_final = LogisticRegression(C=1.0, solver="lbfgs", class_weight="balanced", max_iter=1000)
meta_final.fit(meta_X, y)
blend_oof["meta_lr"] = meta_oof
blend_test["meta_lr"] = meta_final.predict_proba(meta_test_X)[:, 1]

blend_scores = {
    name: roc_auc_score(y, pred)
    for name, pred in blend_oof.items()
}
blend_table = pd.DataFrame({"blend": list(blend_scores.keys()), "oof_auc": list(blend_scores.values())}).sort_values("oof_auc", ascending=False)
display(blend_table)

BEST_BLEND_NAME = blend_table.iloc[0]["blend"]
final_test_pred = blend_test[BEST_BLEND_NAME]
print("Blend selected:", BEST_BLEND_NAME, "AUC:", blend_scores[BEST_BLEND_NAME])
print("Test prediction summary:")
print(pd.Series(final_test_pred).describe())

# %%
# 22. Feature importance

importance = pd.DataFrame({"feature": X.columns})
importance["lgb_gain"] = lgb_importance
importance["xgb_importance"] = xgb_importance

lr_coef_importance = np.zeros(X.shape[1], dtype="float64")
for model in lr_models:
    lr_coef_importance += np.abs(model.named_steps["clf"].coef_[0]) / len(lr_models)
importance["lr_abs_coef"] = lr_coef_importance

for col in ["lgb_gain", "xgb_importance", "lr_abs_coef"]:
    rng = importance[col].max() - importance[col].min()
    importance[f"{col}_norm"] = (importance[col] - importance[col].min()) / (rng + 1e-12)

importance["avg_importance"] = importance[["lgb_gain_norm", "xgb_importance_norm", "lr_abs_coef_norm"]].mean(axis=1)
importance = importance.sort_values("avg_importance", ascending=False).reset_index(drop=True)

print("Top 50 features:")
display(importance.head(50))

plt.figure(figsize=(10, 12))
sns.barplot(data=importance.head(40), x="avg_importance", y="feature")
plt.title("Top 40 Average Feature Importances")
plt.tight_layout()
plt.show()

IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
importance.to_csv(IMPORTANCE_PATH, index=False)
print("Saved feature importance:", IMPORTANCE_PATH)

# I am not dropping these automatically in the final run; this table is for review.
selected_features = importance.loc[
    importance["avg_importance"] >= importance["avg_importance"].quantile(0.15),
    "feature",
].tolist()
print(f"Features above the bottom-15% importance cutoff: {len(selected_features)} / {X.shape[1]}")

# %%
# 23. SHAP check

print("SHAP sample from the first LightGBM fold")
shap_sample = X.sample(min(SHAP_SAMPLE_SIZE, len(X)), random_state=RANDOM_STATE)
explainer = shap.TreeExplainer(lgb_models[0])
shap_values = explainer.shap_values(shap_sample)
if isinstance(shap_values, list):
    shap_values_to_plot = shap_values[1]
else:
    shap_values_to_plot = shap_values

shap.summary_plot(shap_values_to_plot, shap_sample, plot_type="bar", max_display=30, show=False)
plt.title("SHAP Importance - LightGBM Fold 1")
plt.tight_layout()
plt.show()

shap.summary_plot(shap_values_to_plot, shap_sample, max_display=30, show=False)
plt.title("SHAP Beeswarm - LightGBM Fold 1")
plt.tight_layout()
plt.show()

# %%
# 24. Save OOF predictions

oof_diagnostics = pd.DataFrame({
    "uid": train_model["uid"],
    "TARGET": y,
    "oof_lr_baseline": oof_lr_base,
    "oof_lr": oof_lr,
    "oof_lgb": oof_lgb,
    "oof_xgb": oof_xgb,
    **{f"blend_{name}": pred for name, pred in blend_oof.items()},
})
OOF_PATH = OUTPUT_DIR / "oof_diagnostics.csv"
oof_diagnostics.to_csv(OOF_PATH, index=False)
print("OOF file:", OOF_PATH)
display(oof_diagnostics.head())

# %%
# 25. Submission file

submission = sample_submission.copy()
prediction_column = [col for col in submission.columns if col != "uid"][0]

if not np.array_equal(submission["uid"].values, test_flag["uid"].values):
    print("Sample submission uid order differs from test_flag; aligning predictions by uid.")
    pred_by_uid = pd.Series(final_test_pred, index=test_flag["uid"])
    submission[prediction_column] = submission["uid"].map(pred_by_uid).values
else:
    submission[prediction_column] = final_test_pred

submission[prediction_column] = submission[prediction_column].clip(0, 1)

submission_name = f"final_submission_{YOUR_FIRST_LAST_NAME}.csv"
submission_path = OUTPUT_DIR / submission_name
submission.to_csv(submission_path, index=False)

print("Submission saved:", submission_path)
print("Submission column:", prediction_column)
print("Blend used:", BEST_BLEND_NAME)
display(submission.head())
display(submission[prediction_column].describe())

# %%
# 26. Final sanity check

print("Sanity check")
print("Metric:", "ROC-AUC")
print("Columns:", list(submission.columns))
print("Rows:", len(submission), "| expected:", len(sample_submission))
print("Null predictions:", submission[prediction_column].isna().sum())
print("Prediction range:", float(submission[prediction_column].min()), float(submission[prediction_column].max()))
print("File:", submission_path)
