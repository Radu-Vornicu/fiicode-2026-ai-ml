"""
Optimized SOTA Solution for Bank Telemarketing Prediction
Fast version with strategic model selection for maximum impact
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET = "Subscribed"
ID_COL = "id"
FOLDS = 5
SEEDS = [42, 2026, 1337]  # 3 seeds for good stability/speed balance

BASE_CAT = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome",
]

MONTH_ORDER = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Optimized CatBoost params
CB_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 2000,
    "learning_rate": 0.04,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "random_strength": 0.8,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.15,
    "od_type": "Iter",
    "od_wait": 150,
    "allow_writing_files": False,
    "verbose": False,
    "thread_count": -1,
}

CB_BERN_PARAMS = {
    **CB_PARAMS,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
}

# Optimized LightGBM params
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 2000,
    "learning_rate": 0.04,
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
    "force_col_wise": True,
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def locate_file(filename: str) -> Path:
    candidates = [
        Path("/kaggle/input/fiicode-2026-ai-competition") / filename,
        Path(filename),
        Path.cwd() / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {filename}")


def output_path(filename: str) -> Path:
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        return kaggle_working / filename
    return Path.cwd() / filename


def rank_norm(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def finalize_cats(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    for col in cat_cols:
        df[col] = df[col].fillna("missing").astype(str)
    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def build_features_v1(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Minimal but effective feature set"""
    df = df.copy()
    if ID_COL in df.columns:
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
    
    df["age_band"] = pd.cut(
        df["age"], bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"],
    ).astype(str)
    
    cat_cols = BASE_CAT + ["job_education", "contact_month", "age_band"]
    return finalize_cats(df, cat_cols), cat_cols


def build_features_v2(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Advanced feature engineering"""
    df = df.copy()
    if ID_COL in df.columns:
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
    
    # pdays processing
    df["pdays_was_missing"] = (df["pdays"] == -1).astype(np.int8)
    df["pdays_clean"] = df["pdays"].replace(-1, 999).astype(np.int16)
    
    # Log transforms
    df["duration_log1p"] = np.log1p(df["duration"].clip(lower=0))
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
    
    # Binary features
    df["has_any_loan"] = ((housing == "yes") | (loan == "yes")).astype(np.int8)
    df["is_default"] = (default == "yes").astype(np.int8)
    df["was_contacted_before"] = (df["pdays"] != -1).astype(np.int8)
    df["is_cellular"] = (contact == "cellular").astype(np.int8)
    df["balance_negative"] = (df["balance"] < 0).astype(np.int8)
    df["prev_success"] = (poutcome == "success").astype(np.int8)
    
    # Bucketed features
    df["age_bucket"] = pd.cut(
        df["age"], bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"],
    ).astype(str)
    
    df["campaign_bucket"] = pd.cut(
        df["campaign"], bins=[-1, 1, 2, 4, 9, np.inf],
        labels=["1", "2", "3-4", "5-9", "10+"],
    ).astype(str)
    
    pdays_source = df["pdays"].replace(-1, 999)
    df["pdays_bucket"] = pd.cut(
        pdays_source, bins=[-1, 7, 30, 90, 365, np.inf],
        labels=["<=1w", "8-30d", "31-90d", "91-365d", "365d+"],
    ).astype(str)
    df.loc[df["pdays"] == -1, "pdays_bucket"] = "never"
    
    df["duration_bucket"] = pd.cut(
        df["duration"], bins=[-1, 60, 120, 240, 480, np.inf],
        labels=["<=1m", "1-2m", "2-4m", "4-8m", "8m+"],
    ).astype(str)
    
    df["day_bucket"] = pd.cut(
        df["day"], bins=[0, 10, 20, 31],
        labels=["early", "mid", "late"], include_lowest=True,
    ).astype(str)
    
    # Categorical interactions
    df["job_education"] = job + "__" + education
    df["job_marital"] = job + "__" + marital
    df["contact_month"] = contact + "__" + month
    df["poutcome_month"] = poutcome + "__" + month
    df["loan_default"] = loan + "__" + default
    df["contact_day_bucket"] = contact + "__" + df["day_bucket"].astype(str)
    df["history_state"] = np.where(df["previous"] > 0, poutcome + "__seen", "no_previous")
    df["age_bucket_contact"] = df["age_bucket"].astype(str) + "__" + contact
    df["contact_history_state"] = contact + "__" + df["history_state"].astype(str)
    df["month_pdays_bucket"] = month + "__" + df["pdays_bucket"].astype(str)
    df["job_contact"] = job + "__" + contact
    
    cat_cols = BASE_CAT + [
        "age_bucket", "campaign_bucket", "pdays_bucket", "duration_bucket", "day_bucket",
        "job_education", "job_marital", "contact_month", "poutcome_month", "loan_default",
        "contact_day_bucket", "history_state", "age_bucket_contact",
        "contact_history_state", "month_pdays_bucket", "job_contact",
    ]
    
    return finalize_cats(df, cat_cols), cat_cols


def add_target_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y: pd.Series,
    cat_cols: list[str],
    smoothing: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """CV-based target encoding"""
    train_df = train_df.copy()
    test_df = test_df.copy()
    global_mean = y.mean()
    
    for col in cat_cols:
        if col not in train_df.columns:
            continue
        
        te_col = f"{col}_te"
        train_df[te_col] = np.nan
        
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, valid_idx in kf.split(train_df, y):
            fold_train = train_df.iloc[train_idx]
            fold_y = y.iloc[train_idx]
            
            stats = fold_train.groupby(col).apply(
                lambda x: (fold_y.loc[x.index].sum() + smoothing * global_mean) / 
                          (len(x) + smoothing)
            )
            
            train_df.iloc[valid_idx, train_df.columns.get_loc(te_col)] = (
                train_df.iloc[valid_idx][col].map(stats).fillna(global_mean)
            )
        
        stats = train_df.groupby(col).apply(
            lambda x: (y.loc[x.index].sum() + smoothing * global_mean) / 
                      (len(x) + smoothing)
        )
        test_df[te_col] = test_df[col].map(stats).fillna(global_mean)
    
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
    name: str = "CB",
) -> tuple[np.ndarray, np.ndarray]:
    """Train CatBoost ensemble"""
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
        
        seed_auc = roc_auc_score(y, seed_oof)
        print(f"    {name} seed {seed} AUC: {seed_auc:.6f}")
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
    name: str = "LGBM",
) -> tuple[np.ndarray, np.ndarray]:
    """Train LightGBM ensemble"""
    oof = np.zeros(len(x_train), dtype=float)
    test_pred = np.zeros(len(x_test), dtype=float)
    
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
            model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)])
            
            seed_oof[valid_idx] = model.predict_proba(valid_x)[:, 1]
            seed_test += model.predict_proba(x_test_lgb)[:, 1] / FOLDS
        
        seed_auc = roc_auc_score(y, seed_oof)
        print(f"    {name} seed {seed} AUC: {seed_auc:.6f}")
        oof += seed_oof / len(seeds)
        test_pred += seed_test / len(seeds)
    
    return oof, test_pred


# ============================================================================
# BLENDING
# ============================================================================

def grid_search_blend(
    predictions: dict[str, np.ndarray],
    y: pd.Series,
    step: int = 50,
) -> tuple[float, dict[str, float]]:
    """Grid search for optimal blend weights"""
    names = list(predictions.keys())
    n = len(names)
    
    best_auc = -1.0
    best_weights = {}
    
    if n == 2:
        for a in range(step + 1):
            w = [a / step, (step - a) / step]
            blend = sum(w[i] * rank_norm(predictions[names[i]]) for i in range(n))
            auc = roc_auc_score(y, blend)
            if auc > best_auc:
                best_auc = auc
                best_weights = {names[i]: w[i] for i in range(n)}
    
    elif n == 3:
        for a in range(step + 1):
            for b in range(step + 1 - a):
                c = step - a - b
                w = [a / step, b / step, c / step]
                blend = sum(w[i] * rank_norm(predictions[names[i]]) for i in range(n))
                auc = roc_auc_score(y, blend)
                if auc > best_auc:
                    best_auc = auc
                    best_weights = {names[i]: w[i] for i in range(n)}
    
    elif n == 4:
        for a in range(0, step + 1, 5):
            for b in range(0, step + 1 - a, 5):
                for c in range(0, step + 1 - a - b, 5):
                    d = step - a - b - c
                    w = [a / step, b / step, c / step, d / step]
                    blend = sum(w[i] * rank_norm(predictions[names[i]]) for i in range(n))
                    auc = roc_auc_score(y, blend)
                    if auc > best_auc:
                        best_auc = auc
                        best_weights = {names[i]: w[i] for i in range(n)}
    
    else:
        w = 1.0 / n
        best_weights = {name: w for name in names}
        blend = sum(w * rank_norm(predictions[name]) for name in names)
        best_auc = roc_auc_score(y, blend)
    
    return best_auc, best_weights


def apply_blend(
    predictions: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    result = np.zeros(len(next(iter(predictions.values()))), dtype=np.float64)
    for name, preds in predictions.items():
        result += weights.get(name, 0.0) * rank_norm(preds)
    return result


def build_submission(test_ids: pd.Series, preds: np.ndarray, filename: str) -> Path:
    path = output_path(filename)
    submission = pd.DataFrame({
        ID_COL: test_ids,
        TARGET: np.clip(preds, 1e-6, 1.0 - 1e-6),
    })
    submission.to_csv(path, index=False)
    print(f"Saved {filename} -> {path}")
    return path


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 60)
    print("Optimized SOTA Pipeline")
    print("=" * 60)
    
    train = pd.read_csv(locate_file("train.csv"))
    test = pd.read_csv(locate_file("test.csv"))
    y = train[TARGET].astype(int)
    train_features = train.drop(columns=[TARGET])
    
    print(f"\nTrain: {len(train)}, Test: {len(test)}, Target mean: {y.mean():.4f}")
    
    oof_results = {}
    test_results = {}
    
    # =========================================================================
    # MODEL 1: CatBoost with V1 features
    # =========================================================================
    print("\n[1] CatBoost V1 (minimal features)")
    x_train_v1, cat_cols_v1 = build_features_v1(train_features)
    x_test_v1, _ = build_features_v1(test)
    
    cb_v1_oof, cb_v1_test = train_catboost(
        x_train_v1, x_test_v1, y, cat_cols_v1,
        CB_PARAMS, SEEDS, name="CB_V1",
    )
    auc = roc_auc_score(y, cb_v1_oof)
    print(f"  CB_V1 AUC: {auc:.6f}")
    oof_results["cb_v1"] = cb_v1_oof
    test_results["cb_v1"] = cb_v1_test
    
    # =========================================================================
    # MODEL 2: CatBoost with V2 features + class weights
    # =========================================================================
    print("\n[2] CatBoost V2 (advanced features + class weights)")
    x_train_v2, cat_cols_v2 = build_features_v2(train_features)
    x_test_v2, _ = build_features_v2(test)
    
    cb_v2_oof, cb_v2_test = train_catboost(
        x_train_v2, x_test_v2, y, cat_cols_v2,
        CB_PARAMS, SEEDS, use_class_weight=True, name="CB_V2",
    )
    auc = roc_auc_score(y, cb_v2_oof)
    print(f"  CB_V2 AUC: {auc:.6f}")
    oof_results["cb_v2"] = cb_v2_oof
    test_results["cb_v2"] = cb_v2_test
    
    # =========================================================================
    # MODEL 3: CatBoost Bernoulli with V2 features
    # =========================================================================
    print("\n[3] CatBoost Bernoulli (V2 features)")
    cb_bern_oof, cb_bern_test = train_catboost(
        x_train_v2, x_test_v2, y, cat_cols_v2,
        CB_BERN_PARAMS, SEEDS, use_class_weight=True, name="CB_BERN",
    )
    auc = roc_auc_score(y, cb_bern_oof)
    print(f"  CB_BERN AUC: {auc:.6f}")
    oof_results["cb_bern"] = cb_bern_oof
    test_results["cb_bern"] = cb_bern_test
    
    # =========================================================================
    # MODEL 4: LightGBM with V2 features + target encoding
    # =========================================================================
    print("\n[4] LightGBM V2 (with target encoding)")
    te_cols = ["job", "education", "month", "poutcome", "contact", "job_education"]
    x_train_v2_te, x_test_v2_te = add_target_encoding(
        x_train_v2.copy(), x_test_v2.copy(), y, te_cols
    )
    
    lgbm_oof, lgbm_test = train_lightgbm(
        x_train_v2_te, x_test_v2_te, y, cat_cols_v2,
        LGBM_PARAMS, SEEDS, name="LGBM_V2",
    )
    auc = roc_auc_score(y, lgbm_oof)
    print(f"  LGBM_V2 AUC: {auc:.6f}")
    oof_results["lgbm_v2"] = lgbm_oof
    test_results["lgbm_v2"] = lgbm_test
    
    # =========================================================================
    # ENSEMBLE
    # =========================================================================
    print("\n" + "=" * 60)
    print("ENSEMBLE")
    print("=" * 60)
    
    # Find optimal blend
    best_auc, best_weights = grid_search_blend(oof_results, y, step=50)
    print(f"\nBest blend weights: {best_weights}")
    print(f"Best blend AUC: {best_auc:.6f}")
    
    blend_oof = apply_blend(oof_results, best_weights)
    blend_test = apply_blend(test_results, best_weights)
    
    # Simple average as backup
    equal_weights = {k: 1/len(oof_results) for k in oof_results}
    equal_oof = apply_blend(oof_results, equal_weights)
    equal_test = apply_blend(test_results, equal_weights)
    equal_auc = roc_auc_score(y, equal_oof)
    print(f"Equal weight blend AUC: {equal_auc:.6f}")
    
    # =========================================================================
    # SUBMISSIONS
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING SUBMISSIONS")
    print("=" * 60)
    
    # Individual models
    for name, test_pred in test_results.items():
        build_submission(test[ID_COL], rank_norm(test_pred), f"submission_{name}.csv")
    
    # Blend submissions
    build_submission(test[ID_COL], blend_test, "submission_optimized_blend.csv")
    build_submission(test[ID_COL], equal_test, "submission_equal_blend.csv")
    build_submission(test[ID_COL], blend_test, "submission.csv")
    
    print("\n" + "=" * 60)
    print(f"FINAL CV AUC: {best_auc:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
