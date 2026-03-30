from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from submission_layout import submission_output_path


TARGET = "Subscribed"
ID_COL = "id"
SEEDS = [42, 2026]
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
CATBOOST_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 2500,
    "learning_rate": 0.04,
    "depth": 6,
    "l2_leaf_reg": 6.0,
    "random_strength": 1.0,
    "bagging_temperature": 0.2,
    "bootstrap_type": "Bayesian",
    "od_type": "Iter",
    "od_wait": 200,
    "allow_writing_files": False,
    "verbose": False,
}
MODEL_WEIGHTS = {"raw": 0.25, "minimal": 0.50, "curated": 0.25}
PSEUDO_CONFIGS = [
    {
        "name": "pseudo_conservative",
        "quantile": 0.02,
        "pseudo_weight": 0.25,
        "agreement_quantile": 0.85,
    },
    {
        "name": "pseudo_aggressive",
        "quantile": 0.15,
        "pseudo_weight": 1.00,
        "agreement_quantile": 1.00,
    },
]


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


def raw_features(df: pd.DataFrame):
    df = df.copy().drop(columns=[ID_COL])
    for col in BASE_CAT:
        df[col] = df[col].astype(str)
    return df, BASE_CAT.copy()


def minimal_features(df: pd.DataFrame):
    df = df.copy().drop(columns=[ID_COL])
    for col in BASE_CAT:
        df[col] = df[col].astype(str)

    df["month_num"] = df["month"].map(MONTH_ORDER)
    df["has_previous"] = (df["pdays"] != -1).astype(int)
    df["pdays_clean"] = np.where(df["pdays"] == -1, 999, df["pdays"])
    df["balance_log"] = np.sign(df["balance"]) * np.log1p(np.abs(df["balance"]))
    df["duration_log"] = np.log1p(df["duration"])
    df["campaign_log"] = np.log1p(df["campaign"])
    df["previous_log"] = np.log1p(df["previous"])
    df["job_education"] = (df["job"] + "__" + df["education"]).astype(str)
    df["contact_month"] = (df["contact"] + "__" + df["month"]).astype(str)
    df["age_band"] = pd.cut(
        df["age"],
        bins=[17, 25, 35, 45, 55, 65, np.inf],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"],
        include_lowest=True,
    ).astype(str)

    cat_cols = BASE_CAT + ["job_education", "contact_month", "age_band"]
    for col in cat_cols:
        df[col] = df[col].astype(str)
    return df, cat_cols


def curated_features(df: pd.DataFrame):
    df = df.copy().drop(columns=[ID_COL])
    for col in BASE_CAT:
        df[col] = df[col].astype(str)

    df["month_num"] = df["month"].map(MONTH_ORDER)
    df["has_previous"] = (df["pdays"] != -1).astype(int)
    df["pdays_clean"] = np.where(df["pdays"] == -1, 999, df["pdays"])
    df["balance_log"] = np.sign(df["balance"]) * np.log1p(np.abs(df["balance"]))
    df["duration_log"] = np.log1p(df["duration"])
    df["campaign_log"] = np.log1p(df["campaign"])
    df["previous_log"] = np.log1p(df["previous"])
    df["job_education"] = (df["job"] + "__" + df["education"]).astype(str)
    df["contact_month"] = (df["contact"] + "__" + df["month"]).astype(str)
    df["contact_poutcome"] = (df["contact"] + "__" + df["poutcome"]).astype(str)
    df["loan_profile"] = (
        df["default"] + "__" + df["housing"] + "__" + df["loan"]
    ).astype(str)
    df["age_band"] = pd.cut(
        df["age"],
        bins=[17, 25, 35, 45, 55, 65, np.inf],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"],
        include_lowest=True,
    ).astype(str)
    df["duration_band"] = pd.cut(
        df["duration"],
        bins=[0, 60, 120, 240, 480, np.inf],
        labels=["0-60", "61-120", "121-240", "241-480", "480+"],
        include_lowest=True,
    ).astype(str)
    df["balance_band"] = pd.cut(
        df["balance"],
        bins=[-np.inf, 0, 500, 1500, 5000, np.inf],
        labels=["neg", "0-500", "500-1500", "1500-5000", "5000+"],
        include_lowest=True,
    ).astype(str)
    df["pdays_band"] = np.select(
        [df["pdays"] == -1, df["pdays"] <= 7, df["pdays"] <= 30, df["pdays"] <= 90],
        ["never", "1w", "1m", "3m"],
        default="90+",
    )

    cat_cols = BASE_CAT + [
        "job_education",
        "contact_month",
        "contact_poutcome",
        "loan_profile",
        "age_band",
        "duration_band",
        "balance_band",
        "pdays_band",
    ]
    for col in cat_cols:
        df[col] = df[col].astype(str)
    return df, cat_cols


FEATURE_BUILDERS = {
    "raw": raw_features,
    "minimal": minimal_features,
    "curated": curated_features,
}


def fit_seed_ensemble(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y: pd.Series,
    cat_cols: list[str],
    pseudo_x: pd.DataFrame | None = None,
    pseudo_labels: np.ndarray | None = None,
    pseudo_weight: float = 1.0,
):
    model_oof = np.zeros(len(x_train), dtype=float)
    model_test = np.zeros(len(x_test), dtype=float)

    for seed in SEEDS:
        seed_oof = np.zeros(len(x_train), dtype=float)
        seed_test = np.zeros(len(x_test), dtype=float)
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        for train_idx, valid_idx in folds.split(x_train, y):
            train_x = x_train.iloc[train_idx].reset_index(drop=True)
            train_y = y.iloc[train_idx].reset_index(drop=True)
            valid_x = x_train.iloc[valid_idx].reset_index(drop=True)
            valid_y = y.iloc[valid_idx].reset_index(drop=True)

            if pseudo_x is not None and pseudo_labels is not None:
                pseudo_frame = pseudo_x.reset_index(drop=True)
                train_x = pd.concat([train_x, pseudo_frame], axis=0, ignore_index=True)
                train_y = pd.concat(
                    [train_y, pd.Series(pseudo_labels, name=TARGET)], ignore_index=True
                )
                sample_weight = np.concatenate(
                    [
                        np.ones(len(train_idx), dtype=float),
                        np.full(len(pseudo_frame), pseudo_weight, dtype=float),
                    ]
                )
            else:
                sample_weight = None

            model = CatBoostClassifier(**CATBOOST_PARAMS, random_seed=seed)
            model.fit(
                train_x,
                train_y,
                eval_set=(valid_x, valid_y),
                cat_features=cat_cols,
                sample_weight=sample_weight,
                use_best_model=True,
                verbose=False,
            )
            seed_oof[valid_idx] = model.predict_proba(valid_x)[:, 1]
            seed_test += model.predict_proba(x_test)[:, 1] / folds.n_splits

        seed_score = roc_auc_score(y, seed_oof)
        print(f"  seed {seed} AUC: {seed_score:.6f}")
        model_oof += seed_oof / len(SEEDS)
        model_test += seed_test / len(SEEDS)

    return model_oof, model_test


def build_submission(test_ids: pd.Series, preds: np.ndarray, filename: str):
    submission = pd.DataFrame(
        {ID_COL: test_ids, TARGET: np.clip(preds, 1e-6, 1.0 - 1e-6)}
    )
    path = output_path(filename)
    submission.to_csv(path, index=False)
    print(f"Saved {filename} -> {path}")
    return submission


def teacher_score(prob_matrix: np.ndarray) -> np.ndarray:
    mean_prob = prob_matrix.mean(axis=0)
    std_prob = prob_matrix.std(axis=0)
    return np.abs(mean_prob - 0.5) - 0.25 * std_prob


def select_pseudo_labels(
    prob_matrix: np.ndarray,
    quantile: float,
    agreement_quantile: float,
):
    mean_prob = prob_matrix.mean(axis=0)
    std_prob = prob_matrix.std(axis=0)
    score = teacher_score(prob_matrix)
    selected = score >= np.quantile(score, 1.0 - quantile)
    if agreement_quantile < 1.0:
        std_cut = np.quantile(std_prob, agreement_quantile)
        selected &= std_prob <= std_cut
    pseudo_labels = (mean_prob >= 0.5).astype(int)
    return selected, pseudo_labels, mean_prob, std_prob, score


def blend_models(result_dict: dict[str, dict[str, np.ndarray]], key: str):
    blend = np.zeros_like(next(iter(result_dict.values()))[key], dtype=float)
    for model_name, weight in MODEL_WEIGHTS.items():
        blend += weight * rank_norm(result_dict[model_name][key])
    return blend


def main():
    train = pd.read_csv(locate_file("train.csv"))
    test = pd.read_csv(locate_file("test.csv"))
    y = train[TARGET].astype(int)

    print("Training base ensemble")
    base_results = {}
    base_prob_matrix = []
    for model_name, builder in FEATURE_BUILDERS.items():
        print(f"Base model: {model_name}")
        x_train, cat_cols = builder(train.drop(columns=[TARGET]))
        x_test, _ = builder(test)
        oof_pred, test_pred = fit_seed_ensemble(x_train, x_test, y, cat_cols)
        score = roc_auc_score(y, oof_pred)
        base_results[model_name] = {"oof": oof_pred, "test": test_pred, "score": score}
        base_prob_matrix.append(test_pred)
        print(f"  base {model_name} AUC: {score:.6f}")

    base_blend = blend_models(base_results, "test")
    base_oof = blend_models(base_results, "oof")
    print(f"Base blended AUC: {roc_auc_score(y, base_oof):.6f}")
    build_submission(test[ID_COL], base_blend, "submission_lb_base.csv")

    prob_matrix = np.vstack(base_prob_matrix)

    generated = {"base": base_blend}

    for config in PSEUDO_CONFIGS:
        print(
            f"Running {config['name']} "
            f"(q={config['quantile']}, weight={config['pseudo_weight']})"
        )
        selected, pseudo_labels, mean_prob, std_prob, score = select_pseudo_labels(
            prob_matrix,
            quantile=config["quantile"],
            agreement_quantile=config["agreement_quantile"],
        )

        selected_count = int(selected.sum())
        positive_count = int(pseudo_labels[selected].sum())
        negative_count = int(selected_count - positive_count)
        print(
            f"  selected rows: {selected_count}, "
            f"positives: {positive_count}, negatives: {negative_count}"
        )
        print(
            f"  selected mean prob range: "
            f"{mean_prob[selected].min():.4f} to {mean_prob[selected].max():.4f}"
        )
        print(
            f"  selected std range: "
            f"{std_prob[selected].min():.4f} to {std_prob[selected].max():.4f}"
        )

        pseudo_results = {}
        selected_test = test.loc[selected].reset_index(drop=True)
        selected_labels = pseudo_labels[selected]

        for model_name, builder in FEATURE_BUILDERS.items():
            print(f"Pseudo model: {config['name']} / {model_name}")
            x_train, cat_cols = builder(train.drop(columns=[TARGET]))
            x_test, _ = builder(test)
            x_pseudo, _ = builder(selected_test)
            oof_pred, test_pred = fit_seed_ensemble(
                x_train,
                x_test,
                y,
                cat_cols,
                pseudo_x=x_pseudo,
                pseudo_labels=selected_labels,
                pseudo_weight=config["pseudo_weight"],
            )
            score = roc_auc_score(y, oof_pred)
            pseudo_results[model_name] = {
                "oof": oof_pred,
                "test": test_pred,
                "score": score,
            }
            print(f"  pseudo {model_name} AUC: {score:.6f}")

        pseudo_blend = blend_models(pseudo_results, "test")
        pseudo_oof = blend_models(pseudo_results, "oof")
        print(f"  pseudo blended AUC: {roc_auc_score(y, pseudo_oof):.6f}")

        filename = f"submission_{config['name']}.csv"
        build_submission(test[ID_COL], pseudo_blend, filename)
        generated[config["name"]] = pseudo_blend

    hybrid = 0.60 * rank_norm(generated["base"]) + 0.40 * rank_norm(
        generated["pseudo_aggressive"]
    )
    build_submission(test[ID_COL], hybrid, "submission_lb_hybrid.csv")


if __name__ == "__main__":
    main()
