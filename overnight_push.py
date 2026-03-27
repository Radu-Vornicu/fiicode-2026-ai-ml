from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bundle_round2_push import (
    PSEUDO_CONFIG,
    apply_rank_blend,
    build_submission,
    fit_seed_ensemble,
    prepare_minimal_pair,
    prepare_state_pair,
    search_blend,
)
from bundle_inspired_push import BASE_PARAMS, BERN_PARAMS, ID_COL, TARGET, locate_file, rank_norm
from cb_nn_attn10seed import make_nn_features, encode_columns, TabularAttentionNet


CATBOOST_OVERNIGHT_SEEDS = [42, 71, 314, 2026, 3407]
NN_OVERNIGHT_SEEDS = [11, 29, 42, 71, 91, 314, 777, 1234, 2026, 3407]
NN_PSEUDO_WEIGHT = 0.60
NN_EPOCHS = 30
NN_PATIENCE = 5
NN_BATCH_SIZE = 512
NN_BLEND_STEP = 100


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_oof(train_ids: pd.Series, y: pd.Series, columns: dict[str, np.ndarray], filename: str) -> None:
    frame = pd.DataFrame({ID_COL: train_ids, TARGET: y})
    for name, preds in columns.items():
        frame[name] = preds
    path = Path(filename)
    frame.to_csv(path, index=False)
    print(f"Saved {filename} -> {path.resolve()}")


def load_teacher_predictions(test_ids: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    candidates = [
        "submission_public_94650.csv",
        "submission_round2_pseudo_best.csv",
        "submission_round2_meta_best.csv",
    ]
    vectors = []
    for filename in candidates:
        path = Path(filename)
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if len(frame) != len(test_ids):
            continue
        vectors.append(frame[TARGET].to_numpy())

    if not vectors:
        raise FileNotFoundError("No teacher submissions found for overnight pseudo-labeling.")

    prob_matrix = np.vstack(vectors)
    mean_prob = prob_matrix.mean(axis=0)
    std_prob = prob_matrix.std(axis=0)
    score = np.abs(mean_prob - 0.5) - 0.25 * std_prob
    selected = score >= np.quantile(score, 1.0 - PSEUDO_CONFIG["quantile"])
    std_cut = np.quantile(std_prob, PSEUDO_CONFIG["agreement_quantile"])
    selected &= std_prob <= std_cut
    pseudo_labels = (mean_prob >= 0.5).astype(int)
    return selected, pseudo_labels


def fit_attention_fold_pseudo(
    x_train_df: pd.DataFrame,
    x_valid_df: pd.DataFrame,
    x_test_df: pd.DataFrame,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    cat_cols: list[str],
    num_cols: list[str],
    seed: int,
    pseudo_x_df: pd.DataFrame | None = None,
    pseudo_y: np.ndarray | None = None,
    pseudo_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    device = torch.device("cpu")

    if pseudo_x_df is not None and pseudo_y is not None and len(pseudo_x_df) > 0:
        aug_train_df = pd.concat([x_train_df, pseudo_x_df], axis=0, ignore_index=True)
        aug_y = np.concatenate([y_train.astype(np.float32), pseudo_y.astype(np.float32)])
        aug_w = np.concatenate(
            [
                np.ones(len(x_train_df), dtype=np.float32),
                np.full(len(pseudo_x_df), pseudo_weight, dtype=np.float32),
            ]
        )
    else:
        aug_train_df = x_train_df.reset_index(drop=True)
        aug_y = y_train.astype(np.float32)
        aug_w = np.ones(len(x_train_df), dtype=np.float32)

    train_cat, valid_cat, test_cat, cardinalities = encode_columns(
        aug_train_df.reset_index(drop=True),
        x_valid_df.reset_index(drop=True),
        x_test_df.reset_index(drop=True),
        cat_cols,
    )

    scaler = StandardScaler()
    train_num = scaler.fit_transform(aug_train_df[num_cols]).astype(np.float32)
    valid_num = scaler.transform(x_valid_df[num_cols]).astype(np.float32)
    test_num = scaler.transform(x_test_df[num_cols]).astype(np.float32)

    model = TabularAttentionNet(cardinalities, len(num_cols)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    train_dataset = TensorDataset(
        torch.from_numpy(train_cat),
        torch.from_numpy(train_num),
        torch.from_numpy(aug_y),
        torch.from_numpy(aug_w),
    )
    train_loader = DataLoader(train_dataset, batch_size=NN_BATCH_SIZE, shuffle=True)

    valid_cat_t = torch.from_numpy(valid_cat).to(device)
    valid_num_t = torch.from_numpy(valid_num).to(device)
    test_cat_t = torch.from_numpy(test_cat).to(device)
    test_num_t = torch.from_numpy(test_num).to(device)

    best_auc = -1.0
    best_state = None
    patience = 0

    for _ in range(NN_EPOCHS):
        model.train()
        for batch_cat, batch_num, batch_y, batch_w in train_loader:
            optimizer.zero_grad()
            logits = model(batch_cat.to(device), batch_num.to(device))
            losses = loss_fn(logits, batch_y.to(device))
            loss = (losses * batch_w.to(device)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_pred = torch.sigmoid(model(valid_cat_t, valid_num_t)).cpu().numpy()
        valid_auc = roc_auc_score(y_valid, valid_pred)

        if valid_auc > best_auc:
            best_auc = valid_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= NN_PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        valid_pred = torch.sigmoid(model(valid_cat_t, valid_num_t)).cpu().numpy()
        test_pred = torch.sigmoid(model(test_cat_t, test_num_t)).cpu().numpy()
    return valid_pred, test_pred


def fit_attention_ensemble_pseudo(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    y: pd.Series,
    cat_cols: list[str],
    num_cols: list[str],
    pseudo_features: pd.DataFrame | None,
    pseudo_labels: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(train_features), dtype=float)
    counts = np.zeros(len(train_features), dtype=float)
    test_pred = np.zeros(len(test_features), dtype=float)

    total_models = 0
    for seed in NN_OVERNIGHT_SEEDS:
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        seed_oof = np.zeros(len(train_features), dtype=float)
        seed_counts = np.zeros(len(train_features), dtype=float)
        seed_test = np.zeros(len(test_features), dtype=float)

        for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(train_features, y), 1):
            valid_pred, fold_test_pred = fit_attention_fold_pseudo(
                train_features.iloc[train_idx].reset_index(drop=True),
                train_features.iloc[valid_idx].reset_index(drop=True),
                test_features.reset_index(drop=True),
                y.iloc[train_idx].to_numpy(),
                y.iloc[valid_idx].to_numpy(),
                cat_cols,
                num_cols,
                seed=seed * 100 + fold_idx,
                pseudo_x_df=None if pseudo_features is None else pseudo_features.reset_index(drop=True),
                pseudo_y=pseudo_labels,
                pseudo_weight=NN_PSEUDO_WEIGHT,
            )
            seed_oof[valid_idx] += valid_pred
            seed_counts[valid_idx] += 1.0
            seed_test += fold_test_pred / folds.n_splits

        seed_oof /= np.maximum(seed_counts, 1.0)
        seed_auc = roc_auc_score(y, seed_oof)
        print(f"NN seed {seed} AUC: {seed_auc:.6f}")
        oof += seed_oof
        counts += 1.0
        test_pred += seed_test
        total_models += 1

    oof /= np.maximum(counts, 1.0)
    test_pred /= total_models
    return oof, test_pred


def search_two_way_rank_blend(
    left_name: str,
    left_oof: np.ndarray,
    right_name: str,
    right_oof: np.ndarray,
    y: pd.Series,
) -> tuple[float, dict[str, float], np.ndarray]:
    best_auc = -1.0
    best_weights = {}
    best_blend = None
    left_rank = rank_norm(left_oof)
    right_rank = rank_norm(right_oof)

    for a in range(NN_BLEND_STEP + 1):
        left_weight = a / NN_BLEND_STEP
        right_weight = 1.0 - left_weight
        blend = left_weight * left_rank + right_weight * right_rank
        auc = roc_auc_score(y, blend)
        if auc > best_auc:
            best_auc = auc
            best_weights = {left_name: left_weight, right_name: right_weight}
            best_blend = blend

    return best_auc, best_weights, best_blend


def main() -> None:
    torch.set_num_threads(4)

    train = pd.read_csv(locate_file("train.csv"))
    test = pd.read_csv(locate_file("test.csv"))
    y = train[TARGET].astype(int)
    train_features = train.drop(columns=[TARGET])

    selected, teacher_labels = load_teacher_predictions(test[ID_COL])
    selected_test = test.loc[selected].reset_index(drop=True)
    selected_labels = teacher_labels[selected]
    print(
        "Overnight teacher selection: "
        f"{int(selected.sum())} rows, "
        f"{int(selected_labels.sum())} positives, "
        f"{int(len(selected_labels) - selected_labels.sum())} negatives"
    )

    pseudo_views = [
        {
            "name": "pseudo5_minimal_bern",
            "pair_builder": prepare_minimal_pair,
            "params": BERN_PARAMS,
            "use_class_weight": False,
        },
        {
            "name": "pseudo5_state_base_cls",
            "pair_builder": prepare_state_pair,
            "params": BASE_PARAMS,
            "use_class_weight": True,
        },
        {
            "name": "pseudo5_state_bern_cls",
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
            seeds=CATBOOST_OVERNIGHT_SEEDS,
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
    print(f"Overnight pseudo 5-seed weights: {pseudo_weights}")
    print(f"Overnight pseudo 5-seed AUC: {pseudo_auc:.6f}")

    save_oof(
        train[ID_COL],
        y,
        {**pseudo_oof_map, "overnight_pseudo5_best": pseudo_best_oof},
        "oof_overnight_push.csv",
    )
    build_submission(test[ID_COL], pseudo_best_test, "submission_overnight_pseudo5_best.csv")

    print("Training attention NN with teacher pseudo-labels")
    nn_train, nn_cat_cols, nn_num_cols = make_nn_features(train_features)
    nn_test, _, _ = make_nn_features(test)
    nn_pseudo, _, _ = make_nn_features(selected_test)
    nn_oof, nn_test_pred = fit_attention_ensemble_pseudo(
        nn_train,
        nn_test,
        y,
        nn_cat_cols,
        nn_num_cols,
        nn_pseudo,
        selected_labels.astype(np.int64),
    )
    nn_auc = roc_auc_score(y, nn_oof)
    print(f"Overnight pseudo-teacher NN AUC: {nn_auc:.6f}")
    build_submission(test[ID_COL], rank_norm(nn_test_pred), "submission_overnight_nn_teacher.csv")

    safe_submission = pseudo_best_test.copy()
    build_submission(test[ID_COL], safe_submission, "submission_overnight_safe.csv")

    diverse_auc, diverse_weights, _ = search_two_way_rank_blend(
        "catboost_safe",
        pseudo_best_oof,
        "nn_teacher",
        nn_oof,
        y,
    )
    diverse_submission = (
        diverse_weights["catboost_safe"] * rank_norm(pseudo_best_test)
        + diverse_weights["nn_teacher"] * rank_norm(nn_test_pred)
    )
    print(f"Best safe/diverse blend AUC: {diverse_auc:.6f}")
    print(f"Best safe/diverse blend weights: {diverse_weights}")
    build_submission(test[ID_COL], diverse_submission, "submission_overnight_diverse.csv")

    aggressive_submission = 0.75 * rank_norm(pseudo_best_test) + 0.25 * rank_norm(nn_test_pred)
    build_submission(test[ID_COL], aggressive_submission, "submission_overnight_aggressive.csv")

    summary = pd.DataFrame(
        [
            {"name": "overnight_pseudo5_best", "local_auc": pseudo_auc},
            {"name": "overnight_nn_teacher", "local_auc": nn_auc},
            {"name": "overnight_diverse_blend", "local_auc": diverse_auc},
        ]
    )
    summary.to_csv("overnight_summary.csv", index=False)
    print("Saved overnight_summary.csv")


if __name__ == "__main__":
    main()
