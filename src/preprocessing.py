"""Data loading, feature engineering, leakage audit and sklearn preprocessing pipelines.

Leakage rules followed here
---------------------------
* Every transform that learns from data (imputer medians, scaler mean/std, one-hot
  categories, correlation-based feature selection) is fitted on the TRAINING split only.
* Engineered features are row-wise, fixed-formula transforms (no statistics learned
  from the full dataset), so they are safe to compute before the split.
* `customer_id` and `churn` are never used as predictors.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telecom_churn.csv"
TARGET = "churn"
ID_COL = "customer_id"

CATEGORICAL = ["gender", "contract_type", "plan_type", "device_type", "payment_method"]

ENGINEERED = ["spend_per_line", "network_issue_index", "support_intensity",
              "is_new_customer", "n_negative_trends", "bill_shock"]


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def add_engineered_features(df):
    """Row-wise behavioural features with fixed, documented formulas (no learned statistics)."""
    out = df.copy()
    out["spend_per_line"] = out["monthly_charges"] / out["number_of_lines"]
    # composite of drops / outage / call-drop (fixed scale weights, not fitted)
    out["network_issue_index"] = (out["network_drops"] / 5 + out["outage_minutes"] / 60
                                  + out["call_drop_rate"] / 3)
    out["support_intensity"] = out["support_calls"] + 2 * out["complaints"] + 2 * out["tickets_reopened"]
    out["is_new_customer"] = (out["tenure_months"] <= 6).astype(int)
    out["bill_shock"] = (out["bill_change_3m"] > 15).astype(int)
    # count of simultaneous "deterioration" signals (0-5)
    out["n_negative_trends"] = ((out["data_usage_change_3m"] < -15).astype(int)
                                + (out["call_usage_change_3m"] < -15).astype(int)
                                + (out["complaint_change_3m"] > 0).astype(int)
                                + (out["satisfaction_change_3m"] < -1).astype(int)
                                + (out["bill_change_3m"] > 10).astype(int))
    return out


def get_feature_lists(df):
    feats = [c for c in df.columns if c not in (TARGET, ID_COL)]
    cats = [c for c in CATEGORICAL if c in feats]
    nums = [c for c in feats if c not in cats]
    return nums, cats


def get_split(df=None):
    """Reproducible stratified 80/20 split. Returns X_train, X_test, y_train, y_test (ids kept in index)."""
    if df is None:
        df = add_engineered_features(load_data())
    df = df.set_index(ID_COL)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    return train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)


def leakage_audit(X_train, y_train, auc_flag=0.90):
    """Single-feature ROC-AUC on the training set. A feature that alone separates churn
    almost perfectly (AUC >= 0.9) is a leakage red flag. Also checks forbidden column names."""
    forbidden = ["cancel", "terminat", "retention_offer", "churn_date", "latent", "risk_score"]
    bad_names = [c for c in X_train.columns if any(f in c.lower() for f in forbidden)]
    rows = []
    for c in X_train.select_dtypes("number").columns:
        s = X_train[c].fillna(X_train[c].median())
        auc = roc_auc_score(y_train, s)
        rows.append({"feature": c, "single_feature_auc": auc, "strength": abs(auc - 0.5) + 0.5})
    res = pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True)
    res["leakage_flag"] = res["strength"] >= auc_flag
    return res, bad_names


def select_features(X_train, numeric, threshold=0.95):
    """Correlation filter fitted on TRAINING data only: of any pair with |r| > threshold,
    drop the second one. Returns (kept_numeric, dropped_list)."""
    corr = X_train[numeric].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = [c for c in upper.columns if (upper[c] > threshold).any()]
    return [c for c in numeric if c not in dropped], dropped


def build_preprocessor(numeric, categorical, scale=True):
    """ColumnTransformer: numeric -> median impute (+ missing indicator) [+ scale];
    categorical -> most-frequent impute -> one-hot."""
    num_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])


def original_feature_name(transformed_name, categorical):
    """Map a transformed column (e.g. 'cat__contract_type_Two year' or
    'num__missingindicator_nps_score') back to its source column."""
    n = transformed_name.split("__", 1)[1]
    if n.startswith("missingindicator_"):
        return n.replace("missingindicator_", "")
    for c in categorical:
        if n.startswith(c + "_"):
            return c
    return n
