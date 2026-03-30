from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from submission_layout import resolve_submission_path, submission_output_path


TARGET = "Subscribed"
ID_COL = "id"
CB_SEEDS = [42, 2026]
FOLDS = 5
BLEND_STEP = 100  # 1 / 100 = 0.01
BASE_CAT = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]
MONTH_ORDER = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
BASE_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 2500,
    "learning_rate": 0.04,
    "depth": 6,
    "l2_leaf_reg": 6.0,
    "random_strength": 1.0,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.2,
    "od_type": "Iter",
    "od_wait": 200,
    "allow_writing_files": False,
    "verbose": False,
    "thread_count": -1,
}
BERN_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 2500,
    "learning_rate": 0.04,
    "depth": 6,
    "l2_leaf_reg": 6.0,
    "random_strength": 1.0,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.85,
    "od_type": "Iter",
    "od_wait": 200,
    "allow_writing_files": False,
    "verbose": False,
    "thread_count": -1,
}


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
        return submission_output_path(filename, kaggle_working)
    return submission_output_path(filename, Path.cwd())


def rank_norm(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def finalize_cats(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    for col in cat_cols:
        df[col] = df[col].fillna("missing").astype(str)
    return df


def minimal_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy().drop(columns=[ID_COL])
    for col in BASE_CAT:
        df[col] = df[col].astype(str)

    df["month_num"] = df["month"].str.lower().map(MONTH_ORDER).fillna(0).astype(np.int16)
    df["has_previous"] = (df["pdays"] != -1).astype(np.int8)
    df["pdays_clean"] = np.where(df["pdays"] == -1, 999, df["pdays"]).astype(np.int16)
    df["balance_log"] = np.sign(df["balance"]) * np.log1p(np.abs(df["balance"]))
    df["duration_log"] = np.log1p(df["duration"].clip(lower=0))
    df["campaign_log"] = np.log1p(df["campaign"].clip(lower=0))
    df["previous_log"] = np.log1p(df["previous"].clip(lower=0))
    df["job_education"] = df["job"].astype(str) + "__" + df["education"].astype(str)
    df["contact_month"] = df["contact"].astype(str) + "__" + df["month"].astype(str)
    df["age_band"] = pd.cut(
        df["age"],
        bins=[17, 25, 35, 45, 55, 65, np.inf],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"],
        include_lowest=True,
    ).astype(str)

    cat_cols = BASE_CAT + ["job_education", "contact_month", "age_band"]
    return finalize_cats(df, cat_cols), cat_cols


def bundle_state_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy().drop(columns=[ID_COL])
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
    df["pdays_was_missing"] = (df["pdays"] == -1).astype(np.int8)
    df["pdays_clean"] = df["pdays"].replace(-1, 999).astype(np.int16)

    df["duration_log1p"] = np.log1p(df["duration"].clip(lower=0))
    df["balance_log1p"] = np.log1p(df["balance"].clip(lower=0))
    df["balance_abs_log1p"] = np.log1p(df["balance"].abs())
    df["campaign_log1p"] = np.log1p(df["campaign"].clip(lower=0))
    df["previous_log1p"] = np.log1p(df["previous"].clip(lower=0))
    df["pdays_log1p"] = np.log1p(df["pdays_clean"])

    df["contacts_total"] = df["campaign"] + df["previous"]
    df["duration_per_campaign"] = df["duration"] / (df["campaign"] + 1)
    df["balance_per_age"] = df["balance"] / (df["age"] + 1)
    df["previous_per_campaign"] = df["previous"] / (df["campaign"] + 1)
    df["campaign_x_previous"] = df["campaign"] * df["previous"]
    df["duration_x_campaign"] = df["duration"] * df["campaign"]

    df["has_any_loan"] = ((housing == "yes") | (loan == "yes")).astype(np.int8)
    df["is_default"] = (default == "yes").astype(np.int8)
    df["was_contacted_before"] = (df["pdays"] != -1).astype(np.int8)
    df["is_cellular"] = (contact == "cellular").astype(np.int8)

    df["age_bucket"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"],
    ).astype(str)
    df["job_education"] = job + "__" + education
    df["job_marital"] = job + "__" + marital
    df["contact_month"] = contact + "__" + month
    df["poutcome_month"] = poutcome + "__" + month
    df["loan_default"] = loan + "__" + default

    pdays_source = df["pdays"].replace(-1, 999)
    df["balance_signed_log1p"] = np.sign(df["balance"]) * np.log1p(df["balance"].abs())
    df["balance_negative"] = (df["balance"] < 0).astype(np.int8)
    df["balance_nonpositive"] = (df["balance"] <= 0).astype(np.int8)
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
    df["pdays_bucket"] = pd.cut(
        pdays_source,
        bins=[-1, 7, 30, 90, 365, np.inf],
        labels=["<=1w", "8-30d", "31-90d", "91-365d", "365d+"],
    ).astype(str)
    df.loc[df["pdays"] == -1, "pdays_bucket"] = "never"
    df["duration_bucket"] = pd.cut(
        df["duration"],
        bins=[-1, 60, 120, 240, 480, np.inf],
        labels=["<=1m", "1-2m", "2-4m", "4-8m", "8m+"],
    ).astype(str)
    df["day_bucket"] = pd.cut(
        df["day"],
        bins=[0, 10, 20, 31],
        labels=["early", "mid", "late"],
        include_lowest=True,
    ).astype(str)
    df["contact_day_bucket"] = contact + "__" + df["day_bucket"].astype(str)
    df["month_day_bucket"] = month + "__" + df["day_bucket"].astype(str)
    df["history_state"] = np.where(df["previous"] > 0, poutcome + "__seen", "no_previous")

    df["age_bucket_contact"] = df["age_bucket"].astype(str) + "__" + contact
    df["age_bucket_history_state"] = (
        df["age_bucket"].astype(str) + "__" + df["history_state"].astype(str)
    )
    df["loan_housing"] = loan + "__" + housing
    df["contact_history_state"] = contact + "__" + df["history_state"].astype(str)
    df["month_pdays_bucket"] = month + "__" + df["pdays_bucket"].astype(str)
    df["job_contact"] = job + "__" + contact
    df["education_contact"] = education + "__" + contact
    df["housing_default"] = housing + "__" + default
    df["marital_loan"] = marital + "__" + loan

    cat_cols = BASE_CAT + [
        "age_bucket",
        "job_education",
        "job_marital",
        "contact_month",
        "poutcome_month",
        "loan_default",
        "campaign_bucket",
        "previous_bucket",
        "pdays_bucket",
        "duration_bucket",
        "day_bucket",
        "contact_day_bucket",
        "month_day_bucket",
        "history_state",
        "age_bucket_contact",
        "age_bucket_history_state",
        "loan_housing",
        "contact_history_state",
        "month_pdays_bucket",
        "job_contact",
        "education_contact",
        "housing_default",
        "marital_loan",
    ]
    return finalize_cats(df, cat_cols), cat_cols


def fit_seed_ensemble(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y: pd.Series,
    cat_cols: list[str],
    model_params: dict,
    use_class_weight: bool,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(x_train), dtype=float)
    test_pred = np.zeros(len(x_test), dtype=float)

    for seed in CB_SEEDS:
        seed_oof = np.zeros(len(x_train), dtype=float)
        seed_test = np.zeros(len(x_test), dtype=float)
        folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)

        for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(x_train, y), 1):
            train_x = x_train.iloc[train_idx].reset_index(drop=True)
            train_y = y.iloc[train_idx].reset_index(drop=True)
            valid_x = x_train.iloc[valid_idx].reset_index(drop=True)
            valid_y = y.iloc[valid_idx].reset_index(drop=True)

            params = dict(model_params)
            if use_class_weight:
                params["auto_class_weights"] = "Balanced"

            model = CatBoostClassifier(**params, random_seed=seed)
            model.fit(
                train_x,
                train_y,
                eval_set=(valid_x, valid_y),
                cat_features=cat_cols,
                use_best_model=True,
                verbose=False,
            )
            seed_oof[valid_idx] = model.predict_proba(valid_x)[:, 1]
            seed_test += model.predict_proba(x_test)[:, 1] / FOLDS
            fold_auc = roc_auc_score(valid_y, seed_oof[valid_idx])
            print(f"    seed {seed} fold {fold_idx} AUC: {fold_auc:.6f}")

        seed_auc = roc_auc_score(y, seed_oof)
        print(f"  seed {seed} full AUC: {seed_auc:.6f}")
        oof += seed_oof / len(CB_SEEDS)
        test_pred += seed_test / len(CB_SEEDS)

    return oof, test_pred


def build_submission(test_ids: pd.Series, preds: np.ndarray, filename: str) -> Path:
    path = output_path(filename)
    submission = pd.DataFrame(
        {ID_COL: test_ids, TARGET: np.clip(preds, 1e-6, 1.0 - 1e-6)}
    )
    submission.to_csv(path, index=False)
    print(f"Saved {filename} -> {path}")
    return path


def save_oof(train_ids: pd.Series, y: pd.Series, frames: dict[str, np.ndarray], filename: str) -> Path:
    path = output_path(filename)
    oof_frame = pd.DataFrame({ID_COL: train_ids, TARGET: y})
    for name, preds in frames.items():
        oof_frame[name] = preds
    oof_frame.to_csv(path, index=False)
    print(f"Saved {filename} -> {path}")
    return path


def search_blend(
    pred_map: dict[str, np.ndarray],
    y: pd.Series,
    mode: str,
) -> tuple[float, dict[str, float], np.ndarray]:
    names = list(pred_map)
    prepared = {}
    for name, preds in pred_map.items():
        prepared[name] = rank_norm(preds) if mode == "rank" else preds

    best_auc = -1.0
    best_weights = {}
    best_blend = None

    for a in range(BLEND_STEP + 1):
        for b in range(BLEND_STEP + 1 - a):
            c = BLEND_STEP - a - b
            weights = [a / BLEND_STEP, b / BLEND_STEP, c / BLEND_STEP]
            blend = np.zeros(len(y), dtype=float)
            for name, weight in zip(names, weights):
                blend += weight * prepared[name]
            auc = roc_auc_score(y, blend)
            if auc > best_auc:
                best_auc = auc
                best_weights = {name: weight for name, weight in zip(names, weights)}
                best_blend = blend

    return best_auc, best_weights, best_blend


def apply_blend(
    pred_map: dict[str, np.ndarray],
    weights: dict[str, float],
    mode: str,
) -> np.ndarray:
    blend = np.zeros(len(next(iter(pred_map.values()))), dtype=float)
    for name, preds in pred_map.items():
        prepared = rank_norm(preds) if mode == "rank" else preds
        blend += weights[name] * prepared
    return blend


def main() -> None:
    train = pd.read_csv(locate_file("train.csv"))
    test = pd.read_csv(locate_file("test.csv"))
    y = train[TARGET].astype(int)

    views = [
        {
            "name": "minimal_bern",
            "builder": minimal_features,
            "params": BERN_PARAMS,
            "use_class_weight": False,
        },
        {
            "name": "state_base_cls",
            "builder": bundle_state_features,
            "params": BASE_PARAMS,
            "use_class_weight": True,
        },
        {
            "name": "state_bern_cls",
            "builder": bundle_state_features,
            "params": BERN_PARAMS,
            "use_class_weight": True,
        },
    ]

    oof_map: dict[str, np.ndarray] = {}
    test_map: dict[str, np.ndarray] = {}

    for view in views:
        print(f"Training {view['name']}")
        x_train, cat_cols = view["builder"](train.drop(columns=[TARGET]))
        x_test, _ = view["builder"](test)
        oof_pred, test_pred = fit_seed_ensemble(
            x_train,
            x_test,
            y,
            cat_cols,
            model_params=view["params"],
            use_class_weight=view["use_class_weight"],
        )
        auc = roc_auc_score(y, oof_pred)
        oof_map[view["name"]] = oof_pred
        test_map[view["name"]] = test_pred
        print(f"{view['name']} AUC: {auc:.6f}")
        build_submission(test[ID_COL], rank_norm(test_pred), f"submission_{view['name']}.csv")

    equal_weights = {name: 1.0 / len(oof_map) for name in oof_map}
    equal_oof = apply_blend(oof_map, equal_weights, mode="rank")
    equal_test = apply_blend(test_map, equal_weights, mode="rank")
    equal_auc = roc_auc_score(y, equal_oof)
    print(f"Equal rank 3-way blend AUC: {equal_auc:.6f}")
    build_submission(test[ID_COL], equal_test, "submission_bundle_state3way_equal.csv")

    rank_auc, rank_weights, rank_oof = search_blend(oof_map, y, mode="rank")
    prob_auc, prob_weights, prob_oof = search_blend(oof_map, y, mode="prob")
    if rank_auc >= prob_auc:
        best_mode = "rank"
        best_auc = rank_auc
        best_weights = rank_weights
        best_oof = rank_oof
    else:
        best_mode = "prob"
        best_auc = prob_auc
        best_weights = prob_weights
        best_oof = prob_oof

    best_test = apply_blend(test_map, best_weights, mode=best_mode)
    print(f"Best blend mode: {best_mode}")
    print(f"Best blend weights: {best_weights}")
    print(f"Best blended AUC: {best_auc:.6f}")

    save_oof(
        train[ID_COL],
        y,
        {
            **oof_map,
            "bundle_state3way_equal": equal_oof,
            "bundle_state3way_best": best_oof,
        },
        "oof_bundle_inspired.csv",
    )

    build_submission(test[ID_COL], best_test, "submission_bundle_state3way_best.csv")
    build_submission(test[ID_COL], best_test, "submission.csv")

    nn_path = resolve_submission_path("submission_nn_attn10seed.csv")
    if nn_path.exists():
        nn_submission = pd.read_csv(nn_path)
        if list(nn_submission.columns) == [ID_COL, TARGET] and len(nn_submission) == len(test):
            hybrid = 0.80 * rank_norm(best_test) + 0.20 * rank_norm(
                nn_submission[TARGET].to_numpy()
            )
            build_submission(test[ID_COL], hybrid, "submission_bundle_state3way_nn20.csv")
            print("Saved leaderboard-oriented 80/20 hybrid with existing attention NN predictions.")


if __name__ == "__main__":
    main()
