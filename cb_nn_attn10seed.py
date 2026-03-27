import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from leaderboard_push import (
    FEATURE_BUILDERS,
    TARGET,
    locate_file,
    output_path,
    rank_norm,
    fit_seed_ensemble,
)


ID_COL = "id"
NN_WEIGHT = 0.20
CATBOOST_WEIGHT = 0.80
NN_SEEDS = [11, 29, 42, 71, 91, 314, 777, 1234, 2026, 3407]
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

torch.set_num_threads(4)


def make_nn_features(df: pd.DataFrame):
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
    num_cols = [
        "age",
        "balance",
        "day",
        "duration",
        "campaign",
        "previous",
        "month_num",
        "has_previous",
        "pdays_clean",
        "balance_log",
        "duration_log",
        "campaign_log",
        "previous_log",
    ]
    return df, cat_cols, num_cols


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TabularAttentionNet(nn.Module):
    def __init__(
        self,
        cat_cardinalities: list[int],
        num_dim: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality + 1, d_model) for cardinality in cat_cardinalities]
        )
        self.num_proj = nn.Sequential(
            nn.LayerNorm(num_dim),
            nn.Linear(num_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor):
        cat_tokens = torch.stack(
            [emb(x_cat[:, idx]) for idx, emb in enumerate(self.embeddings)], dim=1
        )
        num_token = self.num_proj(x_num).unsqueeze(1)
        cls = self.cls_token.expand(x_cat.size(0), -1, -1)
        tokens = torch.cat([cls, num_token, cat_tokens], dim=1)
        encoded = self.encoder(tokens)
        pooled = torch.cat([encoded[:, 0], encoded[:, 1]], dim=1)
        return self.head(pooled).squeeze(1)


def encode_columns(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: list[str],
):
    train_out = np.zeros((len(train_df), len(cat_cols)), dtype=np.int64)
    valid_out = np.zeros((len(valid_df), len(cat_cols)), dtype=np.int64)
    test_out = np.zeros((len(test_df), len(cat_cols)), dtype=np.int64)
    cardinalities = []

    for idx, col in enumerate(cat_cols):
        categories = (
            pd.Index(train_df[col].astype(str))
            .append(pd.Index(valid_df[col].astype(str)))
            .append(pd.Index(test_df[col].astype(str)))
            .unique()
        )
        mapping = {value: code + 1 for code, value in enumerate(categories)}
        train_out[:, idx] = train_df[col].astype(str).map(mapping).astype(np.int64)
        valid_out[:, idx] = valid_df[col].astype(str).map(mapping).astype(np.int64)
        test_out[:, idx] = test_df[col].astype(str).map(mapping).astype(np.int64)
        cardinalities.append(len(categories) + 1)

    return train_out, valid_out, test_out, cardinalities


def fit_attention_fold(
    x_train_df: pd.DataFrame,
    x_valid_df: pd.DataFrame,
    x_test_df: pd.DataFrame,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    cat_cols: list[str],
    num_cols: list[str],
    seed: int,
):
    set_seed(seed)
    device = torch.device("cpu")

    train_cat, valid_cat, test_cat, cardinalities = encode_columns(
        x_train_df, x_valid_df, x_test_df, cat_cols
    )

    scaler = StandardScaler()
    train_num = scaler.fit_transform(x_train_df[num_cols]).astype(np.float32)
    valid_num = scaler.transform(x_valid_df[num_cols]).astype(np.float32)
    test_num = scaler.transform(x_test_df[num_cols]).astype(np.float32)

    model = TabularAttentionNet(cardinalities, len(num_cols)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    train_dataset = TensorDataset(
        torch.from_numpy(train_cat),
        torch.from_numpy(train_num),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    valid_cat_t = torch.from_numpy(valid_cat).to(device)
    valid_num_t = torch.from_numpy(valid_num).to(device)
    test_cat_t = torch.from_numpy(test_cat).to(device)
    test_num_t = torch.from_numpy(test_num).to(device)

    best_auc = -1.0
    best_state = None
    patience = 0

    for _ in range(30):
        model.train()
        for batch_cat, batch_num, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_cat.to(device), batch_num.to(device))
            loss = loss_fn(logits, batch_y.to(device))
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
            if patience >= 5:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        valid_pred = torch.sigmoid(model(valid_cat_t, valid_num_t)).cpu().numpy()
        test_pred = torch.sigmoid(model(test_cat_t, test_num_t)).cpu().numpy()
    return valid_pred, test_pred


def fit_attention_ensemble(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    y: pd.Series,
    cat_cols: list[str],
    num_cols: list[str],
):
    oof = np.zeros(len(train_features), dtype=float)
    oof_counts = np.zeros(len(train_features), dtype=float)
    test_pred = np.zeros(len(test_features), dtype=float)

    total_models = 0
    for seed in NN_SEEDS:
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        seed_oof = np.zeros(len(train_features), dtype=float)
        seed_counts = np.zeros(len(train_features), dtype=float)
        seed_test = np.zeros(len(test_features), dtype=float)

        for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(train_features, y), 1):
            valid_pred, fold_test_pred = fit_attention_fold(
                train_features.iloc[train_idx].reset_index(drop=True),
                train_features.iloc[valid_idx].reset_index(drop=True),
                test_features.reset_index(drop=True),
                y.iloc[train_idx].to_numpy(),
                y.iloc[valid_idx].to_numpy(),
                cat_cols,
                num_cols,
                seed=seed * 100 + fold_idx,
            )
            seed_oof[valid_idx] += valid_pred
            seed_counts[valid_idx] += 1.0
            seed_test += fold_test_pred / folds.n_splits

        seed_oof /= np.maximum(seed_counts, 1.0)
        seed_auc = roc_auc_score(y, seed_oof)
        print(f"NN seed {seed} AUC: {seed_auc:.6f}")

        oof += seed_oof
        oof_counts += 1.0
        test_pred += seed_test
        total_models += 1

    oof /= np.maximum(oof_counts, 1.0)
    test_pred /= total_models
    return oof, test_pred


def build_submission(test_ids: pd.Series, preds: np.ndarray, filename: str):
    submission = pd.DataFrame(
        {ID_COL: test_ids, TARGET: np.clip(preds, 1e-6, 1.0 - 1e-6)}
    )
    path = output_path(filename)
    submission.to_csv(path, index=False)
    print(f"Saved {filename} -> {path}")
    return submission


def main():
    train = pd.read_csv(locate_file("train.csv"))
    test = pd.read_csv(locate_file("test.csv"))
    y = train[TARGET].astype(int)

    print("Training CatBoost base ensemble")
    cb_results = {}
    for model_name, builder in FEATURE_BUILDERS.items():
        print(f"CatBoost model: {model_name}")
        x_train, cat_cols = builder(train.drop(columns=[TARGET]))
        x_test, _ = builder(test)
        model_oof, model_test = fit_seed_ensemble(x_train, x_test, y, cat_cols)
        score = roc_auc_score(y, model_oof)
        cb_results[model_name] = {"oof": model_oof, "test": model_test, "score": score}
        print(f"  {model_name} AUC: {score:.6f}")

    cb_oof = (
        0.25 * rank_norm(cb_results["raw"]["oof"])
        + 0.50 * rank_norm(cb_results["minimal"]["oof"])
        + 0.25 * rank_norm(cb_results["curated"]["oof"])
    )
    cb_test = (
        0.25 * rank_norm(cb_results["raw"]["test"])
        + 0.50 * rank_norm(cb_results["minimal"]["test"])
        + 0.25 * rank_norm(cb_results["curated"]["test"])
    )
    cb_auc = roc_auc_score(y, cb_oof)
    print(f"CatBoost blend AUC: {cb_auc:.6f}")
    build_submission(test[ID_COL], cb_test, "submission_cb_base_for_nn.csv")

    print("Training attention neural ensemble")
    nn_train, nn_cat_cols, nn_num_cols = make_nn_features(train.drop(columns=[TARGET]))
    nn_test, _, _ = make_nn_features(test)
    nn_oof, nn_test_pred = fit_attention_ensemble(
        nn_train, nn_test, y, nn_cat_cols, nn_num_cols
    )
    nn_auc = roc_auc_score(y, nn_oof)
    print(f"Attention NN AUC: {nn_auc:.6f}")
    build_submission(test[ID_COL], rank_norm(nn_test_pred), "submission_nn_attn10seed.csv")

    blend_oof = CATBOOST_WEIGHT * rank_norm(cb_oof) + NN_WEIGHT * rank_norm(nn_oof)
    blend_test = CATBOOST_WEIGHT * rank_norm(cb_test) + NN_WEIGHT * rank_norm(nn_test_pred)
    blend_auc = roc_auc_score(y, blend_oof)
    print(f"80/20 CatBoost/NN AUC: {blend_auc:.6f}")
    build_submission(test[ID_COL], blend_test, "submission_cb80_nn20_base.csv")

    current_submission_path = Path("submission.csv")
    if current_submission_path.exists():
        current_submission = pd.read_csv(current_submission_path)
        current_blend = CATBOOST_WEIGHT * rank_norm(
            current_submission[TARGET].to_numpy()
        ) + NN_WEIGHT * rank_norm(nn_test_pred)
        build_submission(
            current_submission[ID_COL],
            current_blend,
            "submission_cb80_nn20_current.csv",
        )


if __name__ == "__main__":
    main()
