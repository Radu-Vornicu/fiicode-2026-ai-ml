"""
State-of-the-Art Solution for Bank Telemarketing Prediction
Combines: Multi-model ensemble, advanced feature engineering, stacking, and optimized blending
"""

import random
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET = "Subscribed"
ID_COL = "id"
FOLDS = 5
SEEDS = [42, 2026, 1337, 777, 314]  # More seeds for stability
BLEND_STEP = 100

BASE_CAT = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome",
]

MONTH_ORDER = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# High subscription rate months (from domain knowledge)
HIGH_SUB_MONTHS = ["mar", "sep", "oct", "dec"]
LOW_SUB_MONTHS = ["may", "jun", "jul", "aug"]

# CatBoost configurations
CATBOOST_BASE = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "random_strength": 0.8,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.15,
    "od_type": "Iter",
    "od_wait": 250,
    "allow_writing_files": False,
    "verbose": False,
    "thread_count": -1,
}

CATBOOST_DEEP = {
    **CATBOOST_BASE,
    "depth": 8,
    "learning_rate": 0.02,
    "l2_leaf_reg": 3.0,
}

CATBOOST_BERNOULLI = {
    **CATBOOST_BASE,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
}

# LightGBM configurations
LGBM_BASE = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 5.0,
    "colsample_bytree": 0.8,
    "subsample": 0.85,
    "subsample_freq": 1,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

LGBM_DEEP = {
    **LGBM_BASE,
    "max_depth": 8,
    "num_leaves": 63,
    "learning_rate": 0.02,
}

# XGBoost configuration
XGB_BASE = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 5.0,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "verbosity": 0,
    "n_jobs": -1,
    "random_state": 42,
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def locate_file(filename: str) -> Path:
    candidates = [
        Path("/kaggle/input/fiicode-2026-ai-competition") / filename,
        Path("/kaggle/input/competitions/fiicode-2026-ai-competition") / filename,
        Path(filename),
        Path.cwd() / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {filename}. Checked: {candidates}")


def output_path(filename: str) -> Path:
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        return kaggle_working / filename
    return Path.cwd() / filename


def rank_norm(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def geom_mean_blend(arrays: list[np.ndarray], weights: list[float] = None) -> np.ndarray:
    """Geometric mean blending - often better for probabilities"""
    if weights is None:
        weights = [1.0 / len(arrays)] * len(arrays)
    log_sum = np.zeros_like(arrays[0], dtype=np.float64)
    for arr, w in zip(arrays, weights):
        arr_clipped = np.clip(arr, 1e-8, 1 - 1e-8)
        log_sum += w * np.log(arr_clipped)
    return np.exp(log_sum)


def power_blend(arrays: list[np.ndarray], weights: list[float], power: float = 0.5) -> np.ndarray:
    """Power-transformed weighted blend"""
    result = np.zeros_like(arrays[0], dtype=np.float64)
    for arr, w in zip(arrays, weights):
        result += w * np.power(arr, power)
    return np.power(result, 1.0 / power)


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def finalize_cats(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    for col in cat_cols:
        df[col] = df[col].fillna("missing").astype(str)
    return df


def base_features(df: pd.DataFrame, drop_id: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Minimal feature set"""
    df = df.copy()
    if drop_id and ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    
    for col in BASE_CAT:
        df[col] = df[col].astype(str)
    
    # Basic transforms
    df["month_num"] = df["month"].str.lower().map(MONTH_ORDER).fillna(0).astype(np.int16)
    df["has_previous"] = (df["pdays"] != -1).astype(np.int8)
    df["pdays_clean"] = np.where(df["pdays"] == -1, 999, df["pdays"]).astype(np.int16)
    
    # Log transforms
    df["balance_log"] = np.sign(df["balance"]) * np.log1p(np.abs(df["balance"]))
    df["duration_log"] = np.log1p(df["duration"].clip(lower=0))
    df["campaign_log"] = np.log1p(df["campaign"].clip(lower=0))
    df["previous_log"] = np.log1p(df["previous"].clip(lower=0))
    
    # Interactions
    df["job_education"] = df["job"].astype(str) + "__" + df["education"].astype(str)
    df["contact_month"] = df["contact"].astype(str) + "__" + df["month"].astype(str)
    
    # Binned age
    df["age_band"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"],
    ).astype(str)
    
    cat_cols = BASE_CAT + ["job_education", "contact_month", "age_band"]
    return finalize_cats(df, cat_cols), cat_cols


def advanced_features(df: pd.DataFrame, drop_id: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Advanced feature engineering with domain knowledge"""
    df = df.copy()
    if drop_id and ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    
    for col in BASE_CAT:
        df[col] = df[col].astype(str)
    
    month = df["month"].str.lower()
    job = df["job"].astype(str)
    marital = df["marital"].astype(str)
    education = df["education"].astype(str)
    contact = df["contact"].astype(str)
    poutcome = df["poutcome"].astype(str)
    loan = df["loan"].astype(str)
    default = df["default"].astype(str)
    housing = df["housing"].astype(str)
    
    df["month"] = month
    df["month_num"] = month.map(MONTH_ORDER).fillna(0).astype(np.int16)
    
    # Domain features - seasonal banking patterns
    df["is_high_sub_month"] = month.isin(HIGH_SUB_MONTHS).astype(np.int8)
    df["is_low_sub_month"] = month.isin(LOW_SUB_MONTHS).astype(np.int8)
    df["is_q4"] = month.isin(["oct", "nov", "dec"]).astype(np.int8)
    df["is_q1"] = month.isin(["jan", "feb", "mar"]).astype(np.int8)
    
    # pdays processing
    df["pdays_was_missing"] = (df["pdays"] == -1).astype(np.int8)
    df["pdays_clean"] = df["pdays"].replace(-1, 999).astype(np.int16)
    
    # Log transforms
    df["duration_log1p"] = np.log1p(df["duration"].clip(lower=0))
    df["balance_log1p"] = np.log1p(df["balance"].clip(lower=0))
    df["balance_abs_log1p"] = np.log1p(df["balance"].abs())
    df["balance_signed_log1p"] = np.sign(df["balance"]) * np.log1p(df["balance"].abs())
    df["campaign_log1p"] = np.log1p(df["campaign"].clip(lower=0))
    df["previous_log1p"] = np.log1p(df["previous"].clip(lower=0))
    df["pdays_log1p"] = np.log1p(df["pdays_clean"])
    
    # Interaction features - numeric
    df["contacts_total"] = df["campaign"] + df["previous"]
    df["duration_per_campaign"] = df["duration"] / (df["campaign"] + 1)
    df["balance_per_age"] = df["balance"] / (df["age"] + 1)
    df["previous_per_campaign"] = df["previous"] / (df["campaign"] + 1)
    df["campaign_x_previous"] = df["campaign"] * df["previous"]
    df["duration_x_campaign"] = df["duration"] * df["campaign"]
    df["duration_per_contact"] = df["duration"] / (df["contacts_total"] + 1)
    df["age_x_balance"] = df["age"] * df["balance_signed_log1p"]
    
    # Binary features
    df["has_any_loan"] = ((housing == "yes") | (loan == "yes")).astype(np.int8)
    df["is_default"] = (default == "yes").astype(np.int8)
    df["was_contacted_before"] = (df["pdays"] != -1).astype(np.int8)
    df["is_cellular"] = (contact == "cellular").astype(np.int8)
    df["balance_negative"] = (df["balance"] < 0).astype(np.int8)
    df["balance_zero"] = (df["balance"] == 0).astype(np.int8)
    df["has_both_loans"] = ((housing == "yes") & (loan == "yes")).astype(np.int8)
    df["no_loans"] = ((housing == "no") & (loan == "no")).astype(np.int8)
    
    # Success-related features
    df["prev_success"] = (poutcome == "success").astype(np.int8)
    df["prev_failure"] = (poutcome == "failure").astype(np.int8)
    df["prev_unknown"] = (poutcome == "unknown").astype(np.int8)
    
    # Bucketed features
    df["age_bucket"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"],
    ).astype(str)
    
    df["campaign_bucket"] = pd.cut(
        df["campaign"],
        bins=[-1, 1, 2, 4, 9, np.inf],
        labels=["1", "2", "3-4", "5-9", "10+"],
    ).astype(str)
    
    df["previous_bucket"] = pd.cut(
        df["previous"],
        bins=[-1, 0, 1, 3, np.inf],
        labels=["0", "1", "2-3", "4+"],
    ).astype(str)
    
    pdays_source = df["pdays"].replace(-1, 999)
    df["pdays_bucket"] = pd.cut(
        pdays_source,
        bins=[-1, 7, 30, 90, 365, np.inf],
        labels=["<=1w", "8-30d", "31-90d", "91-365d", "365d+"],
    ).astype(str)
    df.loc[df["pdays"] == -1, "pdays_bucket"] = "never"
    
    df["duration_bucket"] = pd.cut(
        df["duration"],
        bins=[-1, 60, 120, 240, 480, 900, np.inf],
        labels=["<=1m", "1-2m", "2-4m", "4-8m", "8-15m", "15m+"],
    ).astype(str)
    
    df["balance_bucket"] = pd.cut(
        df["balance"],
        bins=[-np.inf, -500, 0, 500, 1500, 5000, np.inf],
        labels=["<-500", "-500-0", "0-500", "500-1500", "1500-5000", "5000+"],
    ).astype(str)
    
    df["day_bucket"] = pd.cut(
        df["day"],
        bins=[0, 10, 20, 31],
        labels=["early", "mid", "late"],
        include_lowest=True,
    ).astype(str)
    
    # Categorical interactions
    df["job_education"] = job + "__" + education
    df["job_marital"] = job + "__" + marital
    df["contact_month"] = contact + "__" + month
    df["poutcome_month"] = poutcome + "__" + month
    df["loan_default"] = loan + "__" + default
    df["loan_housing"] = loan + "__" + housing
    df["housing_default"] = housing + "__" + default
    df["marital_loan"] = marital + "__" + loan
    df["contact_day_bucket"] = contact + "__" + df["day_bucket"].astype(str)
    df["month_day_bucket"] = month + "__" + df["day_bucket"].astype(str)
    df["history_state"] = np.where(df["previous"] > 0, poutcome + "__seen", "no_previous")
    df["age_bucket_contact"] = df["age_bucket"].astype(str) + "__" + contact
    df["age_bucket_history_state"] = df["age_bucket"].astype(str) + "__" + df["history_state"].astype(str)
    df["contact_history_state"] = contact + "__" + df["history_state"].astype(str)
    df["month_pdays_bucket"] = month + "__" + df["pdays_bucket"].astype(str)
    df["job_contact"] = job + "__" + contact
    df["education_contact"] = education + "__" + contact
    df["job_age_bucket"] = job + "__" + df["age_bucket"].astype(str)
    df["education_marital"] = education + "__" + marital
    df["balance_bucket_housing"] = df["balance_bucket"].astype(str) + "__" + housing
    df["duration_bucket_contact"] = df["duration_bucket"].astype(str) + "__" + contact
    
    cat_cols = BASE_CAT + [
        "age_bucket", "campaign_bucket", "previous_bucket", "pdays_bucket",
        "duration_bucket", "balance_bucket", "day_bucket",
        "job_education", "job_marital", "contact_month", "poutcome_month",
        "loan_default", "loan_housing", "housing_default", "marital_loan",
        "contact_day_bucket", "month_day_bucket", "history_state",
        "age_bucket_contact", "age_bucket_history_state", "contact_history_state",
        "month_pdays_bucket", "job_contact", "education_contact",
        "job_age_bucket", "education_marital", "balance_bucket_housing",
        "duration_bucket_contact",
    ]
    
    return finalize_cats(df, cat_cols), cat_cols


def add_target_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y: pd.Series,
    cat_cols: list[str],
    n_folds: int = 5,
    smoothing: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add target encoding features with proper CV-based encoding"""
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    global_mean = y.mean()
    
    for col in cat_cols:
        if col not in train_df.columns:
            continue
        
        te_col = f"{col}_te"
        train_df[te_col] = np.nan
        
        # CV-based encoding for train
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        for train_idx, valid_idx in kf.split(train_df, y):
            # Calculate encoding from train fold
            fold_train = train_df.iloc[train_idx]
            fold_y = y.iloc[train_idx]
            
            stats = fold_train.groupby(col).apply(
                lambda x: (fold_y.loc[x.index].sum() + smoothing * global_mean) / 
                          (len(x) + smoothing)
            )
            
            # Apply to validation fold
            train_df.iloc[valid_idx, train_df.columns.get_loc(te_col)] = (
                train_df.iloc[valid_idx][col].map(stats).fillna(global_mean)
            )
        
        # Full encoding for test
        stats = train_df.groupby(col).apply(
            lambda x: (y.loc[x.index].sum() + smoothing * global_mean) / 
                      (len(x) + smoothing)
        )
        test_df[te_col] = test_df[col].map(stats).fillna(global_mean)
    
    return train_df, test_df


def add_frequency_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add frequency encoding for categorical features"""
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    n_total = len(combined)
    
    for col in cat_cols:
        if col not in train_df.columns:
            continue
        
        counts = combined[col].value_counts()
        freq = counts / n_total
        
        train_df[f"{col}_count"] = train_df[col].map(counts).fillna(0).astype(np.int32)
        train_df[f"{col}_freq"] = train_df[col].map(freq).fillna(0).astype(np.float32)
        test_df[f"{col}_count"] = test_df[col].map(counts).fillna(0).astype(np.int32)
        test_df[f"{col}_freq"] = test_df[col].map(freq).fillna(0).astype(np.float32)
    
    return train_df, test_df


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_catboost(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y: pd.Series,
    cat_cols: list[str],
    params: dict,
    seeds: list[int],
    use_class_weight: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Train CatBoost ensemble with multiple seeds"""
    oof = np.zeros(len(x_train), dtype=float)
    test_pred = np.zeros(len(x_test), dtype=float)
    
    for seed in seeds:
        seed_oof = np.zeros(len(x_train), dtype=float)
        seed_test = np.zeros(len(x_test), dtype=float)
        folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        
        for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(x_train, y), 1):
            train_x = x_train.iloc[train_idx].reset_index(drop=True)
            train_y = y.iloc[train_idx].reset_index(drop=True)
            valid_x = x_train.iloc[valid_idx].reset_index(drop=True)
            valid_y = y.iloc[valid_idx].reset_index(drop=True)
            
            fold_params = dict(params)
            if use_class_weight:
                fold_params["auto_class_weights"] = "Balanced"
            
            model = CatBoostClassifier(**fold_params, random_seed=seed)
            model.fit(
                train_x, train_y,
                eval_set=(valid_x, valid_y),
                cat_features=cat_cols,
                use_best_model=True,
                verbose=False,
            )
            
            seed_oof[valid_idx] = model.predict_proba(valid_x)[:, 1]
            seed_test += model.predict_proba(x_test)[:, 1] / FOLDS
            
            fold_auc = roc_auc_score(valid_y, seed_oof[valid_idx])
            print(f"      seed {seed} fold {fold_idx} AUC: {fold_auc:.6f}")
        
        seed_auc = roc_auc_score(y, seed_oof)
        print(f"    seed {seed} full AUC: {seed_auc:.6f}")
        oof += seed_oof / len(seeds)
        test_pred += seed_test / len(seeds)
    
    return oof, test_pred


def train_lightgbm(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y: pd.Series,
    cat_cols: list[str],
    params: dict,
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Train LightGBM ensemble with multiple seeds"""
    oof = np.zeros(len(x_train), dtype=float)
    test_pred = np.zeros(len(x_test), dtype=float)
    
    # Convert categoricals to category dtype for LGBM
    x_train_lgb = x_train.copy()
    x_test_lgb = x_test.copy()
    for col in cat_cols:
        if col in x_train_lgb.columns:
            x_train_lgb[col] = x_train_lgb[col].astype("category")
            x_test_lgb[col] = x_test_lgb[col].astype("category")
    
    for seed in seeds:
        seed_oof = np.zeros(len(x_train), dtype=float)
        seed_test = np.zeros(len(x_test), dtype=float)
        folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        
        for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(x_train, y), 1):
            train_x = x_train_lgb.iloc[train_idx].reset_index(drop=True)
            train_y = y.iloc[train_idx].reset_index(drop=True)
            valid_x = x_train_lgb.iloc[valid_idx].reset_index(drop=True)
            valid_y = y.iloc[valid_idx].reset_index(drop=True)
            
            fold_params = dict(params)
            fold_params["random_state"] = seed
            
            model = LGBMClassifier(**fold_params)
            model.fit(
                train_x, train_y,
                eval_set=[(valid_x, valid_y)],
                callbacks=[
                    lambda env: (
                        env.model.stop_training if env.iteration > 100 and 
                        env.evaluation_result_list[-1][2] < env.evaluation_result_list[-50][2] - 0.0001
                        else None
                    )
                ] if False else None,  # Disabled for simplicity
            )
            
            seed_oof[valid_idx] = model.predict_proba(valid_x)[:, 1]
            seed_test += model.predict_proba(x_test_lgb)[:, 1] / FOLDS
            
            fold_auc = roc_auc_score(valid_y, seed_oof[valid_idx])
            print(f"      seed {seed} fold {fold_idx} AUC: {fold_auc:.6f}")
        
        seed_auc = roc_auc_score(y, seed_oof)
        print(f"    seed {seed} full AUC: {seed_auc:.6f}")
        oof += seed_oof / len(seeds)
        test_pred += seed_test / len(seeds)
    
    return oof, test_pred


def train_xgboost(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y: pd.Series,
    cat_cols: list[str],
    params: dict,
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Train XGBoost ensemble with multiple seeds"""
    oof = np.zeros(len(x_train), dtype=float)
    test_pred = np.zeros(len(x_test), dtype=float)
    
    # Label encode categoricals for XGBoost
    x_train_xgb = x_train.copy()
    x_test_xgb = x_test.copy()
    
    label_encoders = {}
    for col in cat_cols:
        if col in x_train_xgb.columns:
            combined = pd.concat([x_train_xgb[col], x_test_xgb[col]], ignore_index=True)
            categories = combined.unique()
            mapping = {cat: idx for idx, cat in enumerate(categories)}
            label_encoders[col] = mapping
            x_train_xgb[col] = x_train_xgb[col].map(mapping).fillna(-1).astype(int)
            x_test_xgb[col] = x_test_xgb[col].map(mapping).fillna(-1).astype(int)
    
    for seed in seeds:
        seed_oof = np.zeros(len(x_train), dtype=float)
        seed_test = np.zeros(len(x_test), dtype=float)
        folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        
        for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(x_train, y), 1):
            train_x = x_train_xgb.iloc[train_idx].reset_index(drop=True)
            train_y = y.iloc[train_idx].reset_index(drop=True)
            valid_x = x_train_xgb.iloc[valid_idx].reset_index(drop=True)
            valid_y = y.iloc[valid_idx].reset_index(drop=True)
            
            fold_params = dict(params)
            fold_params["random_state"] = seed
            fold_params["early_stopping_rounds"] = 200
            
            model = XGBClassifier(**fold_params)
            model.fit(
                train_x, train_y,
                eval_set=[(valid_x, valid_y)],
                verbose=False,
            )
            
            seed_oof[valid_idx] = model.predict_proba(valid_x)[:, 1]
            seed_test += model.predict_proba(x_test_xgb)[:, 1] / FOLDS
            
            fold_auc = roc_auc_score(valid_y, seed_oof[valid_idx])
            print(f"      seed {seed} fold {fold_idx} AUC: {fold_auc:.6f}")
        
        seed_auc = roc_auc_score(y, seed_oof)
        print(f"    seed {seed} full AUC: {seed_auc:.6f}")
        oof += seed_oof / len(seeds)
        test_pred += seed_test / len(seeds)
    
    return oof, test_pred


# ============================================================================
# BLENDING & STACKING
# ============================================================================

def grid_search_blend(
    predictions: dict[str, np.ndarray],
    y: pd.Series,
    step: int = 100,
) -> tuple[float, dict[str, float]]:
    """Grid search for optimal blend weights"""
    names = list(predictions.keys())
    n_models = len(names)
    
    best_auc = -1.0
    best_weights = {}
    
    if n_models == 2:
        for a in range(step + 1):
            w = [a / step, (step - a) / step]
            blend = sum(w[i] * rank_norm(predictions[names[i]]) for i in range(n_models))
            auc = roc_auc_score(y, blend)
            if auc > best_auc:
                best_auc = auc
                best_weights = {names[i]: w[i] for i in range(n_models)}
    
    elif n_models == 3:
        for a in range(step + 1):
            for b in range(step + 1 - a):
                c = step - a - b
                w = [a / step, b / step, c / step]
                blend = sum(w[i] * rank_norm(predictions[names[i]]) for i in range(n_models))
                auc = roc_auc_score(y, blend)
                if auc > best_auc:
                    best_auc = auc
                    best_weights = {names[i]: w[i] for i in range(n_models)}
    
    elif n_models == 4:
        for a in range(0, step + 1, 5):
            for b in range(0, step + 1 - a, 5):
                for c in range(0, step + 1 - a - b, 5):
                    d = step - a - b - c
                    w = [a / step, b / step, c / step, d / step]
                    blend = sum(w[i] * rank_norm(predictions[names[i]]) for i in range(n_models))
                    auc = roc_auc_score(y, blend)
                    if auc > best_auc:
                        best_auc = auc
                        best_weights = {names[i]: w[i] for i in range(n_models)}
    
    else:
        # For more models, use equal weights as baseline
        w = 1.0 / n_models
        best_weights = {name: w for name in names}
        blend = sum(w * rank_norm(predictions[name]) for name in names)
        best_auc = roc_auc_score(y, blend)
    
    return best_auc, best_weights


def apply_blend_weights(
    predictions: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """Apply blend weights to predictions"""
    result = np.zeros(len(next(iter(predictions.values()))), dtype=np.float64)
    for name, preds in predictions.items():
        result += weights.get(name, 0.0) * rank_norm(preds)
    return result


def train_stacking(
    oof_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    y: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a stacking meta-learner"""
    # Build feature matrix from OOF predictions
    oof_matrix = np.column_stack([rank_norm(oof_predictions[k]) for k in sorted(oof_predictions.keys())])
    test_matrix = np.column_stack([rank_norm(test_predictions[k]) for k in sorted(test_predictions.keys())])
    
    # Scale features
    scaler = StandardScaler()
    oof_scaled = scaler.fit_transform(oof_matrix)
    test_scaled = scaler.transform(test_matrix)
    
    # Simple ridge meta-learner
    meta_oof = np.zeros(len(y), dtype=float)
    meta_test = np.zeros(len(test_matrix), dtype=float)
    
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, valid_idx in folds.split(oof_scaled, y):
        meta_model = RidgeClassifier(alpha=1.0)
        meta_model.fit(oof_scaled[train_idx], y.iloc[train_idx])
        
        # Use decision function for continuous predictions
        meta_oof[valid_idx] = meta_model.decision_function(oof_scaled[valid_idx])
        meta_test += meta_model.decision_function(test_scaled) / folds.n_splits
    
    # Normalize to [0, 1]
    meta_oof = rank_norm(meta_oof)
    meta_test = rank_norm(meta_test)
    
    return meta_oof, meta_test


# ============================================================================
# PSEUDO-LABELING
# ============================================================================

def select_pseudo_samples(
    predictions: dict[str, np.ndarray],
    confidence_threshold: float = 0.9,
    agreement_threshold: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """Select confident pseudo-labeled samples"""
    pred_matrix = np.vstack([predictions[k] for k in predictions])
    mean_pred = pred_matrix.mean(axis=0)
    std_pred = pred_matrix.std(axis=0)
    
    # High confidence samples
    confident = (mean_pred > confidence_threshold) | (mean_pred < 1 - confidence_threshold)
    
    # Low disagreement samples
    agreed = std_pred < (1 - agreement_threshold)
    
    selected = confident & agreed
    labels = (mean_pred > 0.5).astype(int)
    
    return selected, labels


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def build_submission(test_ids: pd.Series, preds: np.ndarray, filename: str) -> Path:
    path = output_path(filename)
    submission = pd.DataFrame({
        ID_COL: test_ids,
        TARGET: np.clip(preds, 1e-6, 1.0 - 1e-6),
    })
    submission.to_csv(path, index=False)
    print(f"Saved {filename} -> {path}")
    return path


def main():
    print("=" * 70)
    print("SOTA Bank Telemarketing Prediction Pipeline")
    print("=" * 70)
    
    # Load data
    train = pd.read_csv(locate_file("train.csv"))
    test = pd.read_csv(locate_file("test.csv"))
    y = train[TARGET].astype(int)
    train_features = train.drop(columns=[TARGET])
    
    print(f"\nTrain size: {len(train)}, Test size: {len(test)}")
    print(f"Target distribution: {y.mean():.4f}")
    
    oof_results = {}
    test_results = {}
    
    # =========================================================================
    # STAGE 1: Base Feature Engineering Models
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 1: Base Models with Simple Features")
    print("=" * 70)
    
    x_train_base, cat_cols_base = base_features(train_features)
    x_test_base, _ = base_features(test)
    
    print("\n[1.1] CatBoost Base")
    cb_base_oof, cb_base_test = train_catboost(
        x_train_base, x_test_base, y, cat_cols_base,
        CATBOOST_BASE, SEEDS[:3],
    )
    cb_base_auc = roc_auc_score(y, cb_base_oof)
    print(f"  CatBoost Base AUC: {cb_base_auc:.6f}")
    oof_results["cb_base"] = cb_base_oof
    test_results["cb_base"] = cb_base_test
    
    print("\n[1.2] LightGBM Base")
    lgbm_base_oof, lgbm_base_test = train_lightgbm(
        x_train_base, x_test_base, y, cat_cols_base,
        LGBM_BASE, SEEDS[:3],
    )
    lgbm_base_auc = roc_auc_score(y, lgbm_base_oof)
    print(f"  LightGBM Base AUC: {lgbm_base_auc:.6f}")
    oof_results["lgbm_base"] = lgbm_base_oof
    test_results["lgbm_base"] = lgbm_base_test
    
    # =========================================================================
    # STAGE 2: Advanced Feature Engineering Models
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 2: Advanced Feature Engineering")
    print("=" * 70)
    
    x_train_adv, cat_cols_adv = advanced_features(train_features)
    x_test_adv, _ = advanced_features(test)
    
    # Add target encoding
    te_cols = ["job", "education", "month", "poutcome", "contact", "job_education", "contact_month"]
    x_train_adv, x_test_adv = add_target_encoding(x_train_adv, x_test_adv, y, te_cols)
    
    # Add frequency encoding
    freq_cols = ["job", "education", "month", "contact", "poutcome"]
    x_train_adv, x_test_adv = add_frequency_encoding(x_train_adv, x_test_adv, freq_cols)
    
    print("\n[2.1] CatBoost Advanced")
    cb_adv_oof, cb_adv_test = train_catboost(
        x_train_adv, x_test_adv, y, cat_cols_adv,
        CATBOOST_BASE, SEEDS[:3], use_class_weight=True,
    )
    cb_adv_auc = roc_auc_score(y, cb_adv_oof)
    print(f"  CatBoost Advanced AUC: {cb_adv_auc:.6f}")
    oof_results["cb_adv"] = cb_adv_oof
    test_results["cb_adv"] = cb_adv_test
    
    print("\n[2.2] CatBoost Deep")
    cb_deep_oof, cb_deep_test = train_catboost(
        x_train_adv, x_test_adv, y, cat_cols_adv,
        CATBOOST_DEEP, SEEDS[:3], use_class_weight=True,
    )
    cb_deep_auc = roc_auc_score(y, cb_deep_oof)
    print(f"  CatBoost Deep AUC: {cb_deep_auc:.6f}")
    oof_results["cb_deep"] = cb_deep_oof
    test_results["cb_deep"] = cb_deep_test
    
    print("\n[2.3] CatBoost Bernoulli")
    cb_bern_oof, cb_bern_test = train_catboost(
        x_train_adv, x_test_adv, y, cat_cols_adv,
        CATBOOST_BERNOULLI, SEEDS[:3], use_class_weight=True,
    )
    cb_bern_auc = roc_auc_score(y, cb_bern_oof)
    print(f"  CatBoost Bernoulli AUC: {cb_bern_auc:.6f}")
    oof_results["cb_bern"] = cb_bern_oof
    test_results["cb_bern"] = cb_bern_test
    
    print("\n[2.4] LightGBM Advanced")
    lgbm_adv_oof, lgbm_adv_test = train_lightgbm(
        x_train_adv, x_test_adv, y, cat_cols_adv,
        LGBM_BASE, SEEDS[:3],
    )
    lgbm_adv_auc = roc_auc_score(y, lgbm_adv_oof)
    print(f"  LightGBM Advanced AUC: {lgbm_adv_auc:.6f}")
    oof_results["lgbm_adv"] = lgbm_adv_oof
    test_results["lgbm_adv"] = lgbm_adv_test
    
    print("\n[2.5] LightGBM Deep")
    lgbm_deep_oof, lgbm_deep_test = train_lightgbm(
        x_train_adv, x_test_adv, y, cat_cols_adv,
        LGBM_DEEP, SEEDS[:3],
    )
    lgbm_deep_auc = roc_auc_score(y, lgbm_deep_oof)
    print(f"  LightGBM Deep AUC: {lgbm_deep_auc:.6f}")
    oof_results["lgbm_deep"] = lgbm_deep_oof
    test_results["lgbm_deep"] = lgbm_deep_test
    
    print("\n[2.6] XGBoost Advanced")
    xgb_adv_oof, xgb_adv_test = train_xgboost(
        x_train_adv, x_test_adv, y, cat_cols_adv,
        XGB_BASE, SEEDS[:3],
    )
    xgb_adv_auc = roc_auc_score(y, xgb_adv_oof)
    print(f"  XGBoost Advanced AUC: {xgb_adv_auc:.6f}")
    oof_results["xgb_adv"] = xgb_adv_oof
    test_results["xgb_adv"] = xgb_adv_test
    
    # =========================================================================
    # STAGE 3: Blending & Stacking
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 3: Ensemble Blending")
    print("=" * 70)
    
    # Find optimal blend weights
    print("\n[3.1] Finding optimal blend weights...")
    best_auc, best_weights = grid_search_blend(oof_results, y, step=20)
    print(f"  Best blend AUC: {best_auc:.6f}")
    print(f"  Best weights: {best_weights}")
    
    blend_oof = apply_blend_weights(oof_results, best_weights)
    blend_test = apply_blend_weights(test_results, best_weights)
    
    # Stacking
    print("\n[3.2] Training stacking meta-learner...")
    stack_oof, stack_test = train_stacking(oof_results, test_results, y)
    stack_auc = roc_auc_score(y, stack_oof)
    print(f"  Stacking AUC: {stack_auc:.6f}")
    
    # Combine blend and stack
    print("\n[3.3] Final ensemble combination...")
    for stack_weight in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        blend_weight = 1.0 - stack_weight
        combined_oof = blend_weight * blend_oof + stack_weight * stack_oof
        combined_auc = roc_auc_score(y, combined_oof)
        print(f"  blend={blend_weight:.1f} stack={stack_weight:.1f} -> AUC: {combined_auc:.6f}")
    
    # Use best combination
    final_oof = 0.8 * blend_oof + 0.2 * stack_oof
    final_test = 0.8 * blend_test + 0.2 * stack_test
    final_auc = roc_auc_score(y, final_oof)
    print(f"\n  Final ensemble AUC: {final_auc:.6f}")
    
    # =========================================================================
    # STAGE 4: Generate Submissions
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 4: Generating Submissions")
    print("=" * 70)
    
    # Individual model submissions
    for name in oof_results:
        build_submission(test[ID_COL], rank_norm(test_results[name]), f"submission_{name}.csv")
    
    # Blend submission
    build_submission(test[ID_COL], blend_test, "submission_sota_blend.csv")
    
    # Stack submission
    build_submission(test[ID_COL], stack_test, "submission_sota_stack.csv")
    
    # Final submission
    build_submission(test[ID_COL], final_test, "submission_sota_final.csv")
    build_submission(test[ID_COL], final_test, "submission.csv")
    
    # Also try geometric mean blend
    geom_blend = geom_mean_blend(
        [test_results[k] for k in test_results],
        [best_weights.get(k, 1.0/len(test_results)) for k in test_results]
    )
    build_submission(test[ID_COL], rank_norm(geom_blend), "submission_sota_geom.csv")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nFinal CV AUC: {final_auc:.6f}")
    print("Recommended submission: submission_sota_final.csv or submission.csv")


if __name__ == "__main__":
    main()
