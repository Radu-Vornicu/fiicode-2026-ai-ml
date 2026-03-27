from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from bundle_inspired_push import (
    BASE_PARAMS,
    BERN_PARAMS,
    ID_COL,
    TARGET,
    bundle_state_features,
    locate_file,
    minimal_features,
    output_path,
    rank_norm,
)


BASE_VIEW_WEIGHTS = {
    "minimal_bern": 0.44,
    "state_base_cls": 0.37,
    "state_bern_cls": 0.19,
}
BASE_SEEDS = [42, 2026]
FOLDS = 5
BLEND_STEP = 100
PSEUDO_CONFIG = {
    "quantile": 0.12,
    "agreement_quantile": 0.90,
    "pseudo_weight": 0.75,
}
CATFREQ_COLUMNS = [
    "job",
    "education",
    "contact",
    "month",
    "poutcome",
    "marital",
    "age_bucket",
    "campaign_bucket",
    "pdays_bucket",
    "duration_bucket",
    "history_state",
    "job_education",
    "contact_month",
    "poutcome_month",
]


def build_submission(test_ids: pd.Series, preds: np.ndarray, filename: str) -> Path:
    path = output_path(filename)
    submission = pd.DataFrame(
        {ID_COL: test_ids, TARGET: np.clip(preds, 1e-6, 1.0 - 1e-6)}
    )
    submission.to_csv(path, index=False)
    print(f"Saved {filename} -> {path}")
    return path


def apply_category_frequency_features(
    train_fe: pd.DataFrame,
    test_fe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_fe = train_fe.copy()
    test_fe = test_fe.copy()
    combined = pd.concat([train_fe, test_fe], axis=0, ignore_index=True)
    n_combined = len(combined)

    for col in CATFREQ_COLUMNS:
        if col not in train_fe.columns:
            continue
        col_values = combined[col].fillna("missing").astype(str)
        counts = col_values.value_counts(dropna=False)
        frequencies = counts / n_combined

        train_col = train_fe[col].fillna("missing").astype(str)
        test_col = test_fe[col].fillna("missing").astype(str)
        train_fe[f"{col}_freq"] = train_col.map(frequencies).fillna(0.0).astype(np.float32)
        train_fe[f"{col}_count"] = train_col.map(counts).fillna(0).astype(np.int32)
        train_fe[f"{col}_log_count"] = np.log1p(train_fe[f"{col}_count"]).astype(np.float32)
        test_fe[f"{col}_freq"] = test_col.map(frequencies).fillna(0.0).astype(np.float32)
        test_fe[f"{col}_count"] = test_col.map(counts).fillna(0).astype(np.int32)
        test_fe[f"{col}_log_count"] = np.log1p(test_fe[f"{col}_count"]).astype(np.float32)

    return train_fe, test_fe


def prepare_minimal_pair(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    x_train, cat_cols = minimal_features(train_features)
    x_test, _ = minimal_features(test_features)
    return x_train, x_test, cat_cols


def prepare_state_pair(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    x_train, cat_cols = bundle_state_features(train_features)
    x_test, _ = bundle_state_features(test_features)
    return x_train, x_test, cat_cols


def prepare_state_catfreq_pair(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    x_train, cat_cols = bundle_state_features(train_features)
    x_test, _ = bundle_state_features(test_features)
    x_train, x_test = apply_category_frequency_features(x_train, x_test)
    return x_train, x_test, cat_cols


def fit_seed_ensemble(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y: pd.Series,
    cat_cols: list[str],
    model_params: dict,
    use_class_weight: bool,
    seeds: list[int],
    pseudo_x: pd.DataFrame | None = None,
    pseudo_labels: np.ndarray | None = None,
    pseudo_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
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

            sample_weight = None
            if pseudo_x is not None and pseudo_labels is not None:
                pseudo_frame = pseudo_x.reset_index(drop=True)
                train_x = pd.concat([train_x, pseudo_frame], axis=0, ignore_index=True)
                train_y = pd.concat(
                    [train_y, pd.Series(pseudo_labels, name=TARGET)],
                    axis=0,
                    ignore_index=True,
                )
                sample_weight = np.concatenate(
                    [
                        np.ones(len(train_idx), dtype=float),
                        np.full(len(pseudo_frame), pseudo_weight, dtype=float),
                    ]
                )

            params = dict(model_params)
            if use_class_weight:
                params["auto_class_weights"] = "Balanced"

            model = CatBoostClassifier(**params, random_seed=seed)
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
            seed_test += model.predict_proba(x_test)[:, 1] / FOLDS
            fold_auc = roc_auc_score(valid_y, seed_oof[valid_idx])
            print(f"    seed {seed} fold {fold_idx} AUC: {fold_auc:.6f}")

        seed_auc = roc_auc_score(y, seed_oof)
        print(f"  seed {seed} full AUC: {seed_auc:.6f}")
        oof += seed_oof / len(seeds)
        test_pred += seed_test / len(seeds)

    return oof, test_pred


def search_blend(
    pred_map: dict[str, np.ndarray],
    y: pd.Series,
) -> tuple[float, dict[str, float], np.ndarray]:
    names = list(pred_map)
    ranked = {name: rank_norm(preds) for name, preds in pred_map.items()}

    best_auc = -1.0
    best_weights = {}
    best_blend = None

    if len(names) == 3:
        for a in range(BLEND_STEP + 1):
            for b in range(BLEND_STEP + 1 - a):
                c = BLEND_STEP - a - b
                weights = [a / BLEND_STEP, b / BLEND_STEP, c / BLEND_STEP]
                blend = np.zeros(len(y), dtype=float)
                for name, weight in zip(names, weights):
                    blend += weight * ranked[name]
                auc = roc_auc_score(y, blend)
                if auc > best_auc:
                    best_auc = auc
                    best_weights = {name: weight for name, weight in zip(names, weights)}
                    best_blend = blend
    elif len(names) == 2:
        for a in range(BLEND_STEP + 1):
            b = BLEND_STEP - a
            weights = [a / BLEND_STEP, b / BLEND_STEP]
            blend = weights[0] * ranked[names[0]] + weights[1] * ranked[names[1]]
            auc = roc_auc_score(y, blend)
            if auc > best_auc:
                best_auc = auc
                best_weights = {name: weight for name, weight in zip(names, weights)}
                best_blend = blend
    else:
        raise ValueError("Only 2-model or 3-model blend search is implemented.")

    return best_auc, best_weights, best_blend


def apply_rank_blend(
    pred_map: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    blend = np.zeros(len(next(iter(pred_map.values()))), dtype=float)
    for name, preds in pred_map.items():
        blend += weights[name] * rank_norm(preds)
    return blend


def load_base_predictions() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    oof_frame = pd.read_csv("oof_bundle_inspired.csv")
    oof_map = {name: oof_frame[name].to_numpy() for name in BASE_VIEW_WEIGHTS}
    test_map = {
        name: pd.read_csv(f"submission_{name}.csv")[TARGET].to_numpy()
        for name in BASE_VIEW_WEIGHTS
    }
    return oof_map, test_map


def select_pseudo_rows(test_map: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    ordered_names = list(BASE_VIEW_WEIGHTS)
    matrix = np.vstack([test_map[name] for name in ordered_names])
    weighted_mean = np.zeros(matrix.shape[1], dtype=float)
    for name in ordered_names:
        weighted_mean += BASE_VIEW_WEIGHTS[name] * test_map[name]
    std = matrix.std(axis=0)
    score = np.abs(weighted_mean - 0.5) - 0.25 * std
    selected = score >= np.quantile(score, 1.0 - PSEUDO_CONFIG["quantile"])
    std_cut = np.quantile(std, PSEUDO_CONFIG["agreement_quantile"])
    selected &= std <= std_cut
    pseudo_labels = (weighted_mean >= 0.5).astype(int)
    return selected, pseudo_labels


def main() -> None:
    train = pd.read_csv(locate_file("train.csv"))
    test = pd.read_csv(locate_file("test.csv"))
    y = train[TARGET].astype(int)
    train_features = train.drop(columns=[TARGET])

    base_oof_map, base_test_map = load_base_predictions()
    base_best_oof = apply_rank_blend(base_oof_map, BASE_VIEW_WEIGHTS)
    base_best_test = apply_rank_blend(base_test_map, BASE_VIEW_WEIGHTS)
    base_best_auc = roc_auc_score(y, base_best_oof)
    print(f"Incumbent base blend AUC: {base_best_auc:.6f}")

    selected, pseudo_labels = select_pseudo_rows(base_test_map)
    selected_test = test.loc[selected].reset_index(drop=True)
    selected_labels = pseudo_labels[selected]
    print(
        "Pseudo-label selection: "
        f"{int(selected.sum())} rows, "
        f"{int(selected_labels.sum())} positives, "
        f"{int(len(selected_labels) - selected_labels.sum())} negatives"
    )

    pseudo_views = [
        {
            "name": "pseudo_minimal_bern",
            "pair_builder": prepare_minimal_pair,
            "params": BERN_PARAMS,
            "use_class_weight": False,
        },
        {
            "name": "pseudo_state_base_cls",
            "pair_builder": prepare_state_pair,
            "params": BASE_PARAMS,
            "use_class_weight": True,
        },
        {
            "name": "pseudo_state_bern_cls",
            "pair_builder": prepare_state_pair,
            "params": BERN_PARAMS,
            "use_class_weight": True,
        },
    ]

    pseudo_oof_map: dict[str, np.ndarray] = {}
    pseudo_test_map: dict[str, np.ndarray] = {}

    for view in pseudo_views:
        print(f"Training {view['name']}")
        x_train, x_test, cat_cols = view["pair_builder"](train_features, test)
        x_pseudo, _, _ = view["pair_builder"](selected_test, selected_test)
        oof_pred, test_pred = fit_seed_ensemble(
            x_train,
            x_test,
            y,
            cat_cols,
            model_params=view["params"],
            use_class_weight=view["use_class_weight"],
            seeds=BASE_SEEDS,
            pseudo_x=x_pseudo,
            pseudo_labels=selected_labels,
            pseudo_weight=PSEUDO_CONFIG["pseudo_weight"],
        )
        auc = roc_auc_score(y, oof_pred)
        pseudo_oof_map[view["name"]] = oof_pred
        pseudo_test_map[view["name"]] = test_pred
        print(f"{view['name']} AUC: {auc:.6f}")
        build_submission(test[ID_COL], rank_norm(test_pred), f"submission_{view['name']}.csv")

    pseudo_auc, pseudo_weights, pseudo_best_oof = search_blend(pseudo_oof_map, y)
    pseudo_best_test = apply_rank_blend(pseudo_test_map, pseudo_weights)
    print(f"Best pseudo blend weights: {pseudo_weights}")
    print(f"Best pseudo blend AUC: {pseudo_auc:.6f}")
    build_submission(test[ID_COL], pseudo_best_test, "submission_round2_pseudo_best.csv")

    print("Training state_catfreq_bern_cls")
    x_train_catfreq, x_test_catfreq, cat_cols_catfreq = prepare_state_catfreq_pair(
        train_features,
        test,
    )
    catfreq_oof, catfreq_test = fit_seed_ensemble(
        x_train_catfreq,
        x_test_catfreq,
        y,
        cat_cols_catfreq,
        model_params=BERN_PARAMS,
        use_class_weight=True,
        seeds=BASE_SEEDS,
    )
    catfreq_auc = roc_auc_score(y, catfreq_oof)
    print(f"state_catfreq_bern_cls AUC: {catfreq_auc:.6f}")
    build_submission(test[ID_COL], rank_norm(catfreq_test), "submission_state_catfreq_bern_cls.csv")

    meta_oof_map = {
        "base_best": base_best_oof,
        "pseudo_best": pseudo_best_oof,
        "state_catfreq_bern_cls": catfreq_oof,
    }
    meta_test_map = {
        "base_best": base_best_test,
        "pseudo_best": pseudo_best_test,
        "state_catfreq_bern_cls": catfreq_test,
    }
    meta_auc, meta_weights, meta_best_oof = search_blend(meta_oof_map, y)
    meta_best_test = apply_rank_blend(meta_test_map, meta_weights)
    print(f"Best meta blend weights: {meta_weights}")
    print(f"Best meta blend AUC: {meta_auc:.6f}")
    build_submission(test[ID_COL], meta_best_test, "submission_round2_meta_best.csv")

    nn_path = Path("submission_nn_attn10seed.csv")
    if nn_path.exists():
        nn_pred = pd.read_csv(nn_path)[TARGET].to_numpy()
        meta_nn = 0.80 * rank_norm(meta_best_test) + 0.20 * rank_norm(nn_pred)
        build_submission(test[ID_COL], meta_nn, "submission_round2_meta_nn20.csv")


if __name__ == "__main__":
    main()
