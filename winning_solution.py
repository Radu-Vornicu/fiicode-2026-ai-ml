"""
Winning Solution Implementation - FiiCode 2026 AI Competition
==============================================================
Based on the top proven 0.94839 LB approach:
- CatBoost with blend_buckets features (exp012 style)
- Attention-based Neural Network (exp026 style)  
- 80/20 rank-based blend

Run on Kaggle with GPU enabled for best results.
Expected CV: ~0.937+, Expected LB: 0.948+
"""

import warnings
warnings.filterwarnings('ignore')

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from submission_layout import submission_output_path

# ============================================================================
# CONFIGURATION (from exp012 + exp026)
# ============================================================================

TARGET = 'Subscribed'
ID_COL = 'id'

# Seeds used in winning solution
CATBOOST_SEEDS = [42, 2024, 3407]
NN_SEEDS = [42, 2024, 3407, 777, 1337, 1001, 2718]  # 7 seeds for NN diversity
FOLDS = 5

# Month mapping
MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
}

# CatBoost params from exp012 Optuna tuning
CATBOOST_PARAMS = {
    'iterations': 3500,
    'learning_rate': 0.01377,
    'depth': 6,
    'l2_leaf_reg': 3.976,
    'random_strength': 1.084,
    'bagging_temperature': 0.733,
    'bootstrap_type': 'Bayesian',
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'allow_writing_files': False,
    'verbose': False,
    'thread_count': -1,
}

# Attention NN params from exp026
NN_PARAMS = {
    'emb_dim': 24,
    'hidden_dim': 192,
    'n_layers': 4,
    'heads': 6,
    'dropout': 0.2,
    'epochs': 90,
    'batch_size': 1024,
    'lr': 0.001,
    'weight_decay': 0.01,
    'patience': 16,
}

# Base categorical columns
BASE_CATEGORICAL = [
    'job', 'marital', 'education', 'default', 'housing',
    'loan', 'contact', 'month', 'poutcome'
]


def locate_data():
    """Find train/test CSV files"""
    candidates = [
        Path('/kaggle/input/fiicode-2026-ai-competition'),
        Path('.'),
        Path.cwd(),
    ]
    for path in candidates:
        if (path / 'train.csv').exists():
            return path
    raise FileNotFoundError('Could not locate train.csv')


def rank_normalize(values):
    """Convert to percentile ranks (0-1)"""
    return pd.Series(values).rank(method='average', pct=True).to_numpy()


def _string_col(data, column, lowercase=False):
    """Safe string extraction with missing handling"""
    values = data[column].fillna('missing').astype(str)
    return values.str.lower() if lowercase else values


# ============================================================================
# FEATURE ENGINEERING - blend_buckets (proven best feature set)
# ============================================================================

def build_blend_bucket_features(data):
    """
    Implements the blend_buckets feature set from the top winning solution.
    This feature engineering approach achieved the best CV and LB scores.
    """
    df = data.copy()
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    
    # Extract string columns
    month = _string_col(df, 'month', lowercase=True)
    job = _string_col(df, 'job')
    marital = _string_col(df, 'marital')
    education = _string_col(df, 'education')
    contact = _string_col(df, 'contact')
    poutcome = _string_col(df, 'poutcome')
    loan = _string_col(df, 'loan')
    default = _string_col(df, 'default')
    housing = _string_col(df, 'housing')
    
    # Basic transformations
    df['month'] = month
    df['month_num'] = month.map(MONTH_MAP).fillna(0).astype(np.int16)
    df['pdays_was_missing'] = (df['pdays'] == -1).astype(np.int8)
    df['pdays_clean'] = df['pdays'].replace(-1, 999)
    
    # Log transformations (handle edge cases)
    df['duration_log1p'] = np.log1p(df['duration'].clip(lower=0))
    df['balance_log1p'] = np.log1p(df['balance'].clip(lower=0))
    df['balance_abs_log1p'] = np.log1p(df['balance'].abs())
    df['campaign_log1p'] = np.log1p(df['campaign'].clip(lower=0))
    df['previous_log1p'] = np.log1p(df['previous'].clip(lower=0))
    df['pdays_log1p'] = np.log1p(df['pdays_clean'])
    
    # Interaction features
    df['contacts_total'] = df['campaign'] + df['previous']
    df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1)
    df['balance_per_age'] = df['balance'] / (df['age'] + 1)
    df['previous_per_campaign'] = df['previous'] / (df['campaign'] + 1)
    df['campaign_x_previous'] = df['campaign'] * df['previous']
    df['duration_x_campaign'] = df['duration'] * df['campaign']
    
    # Binary indicators
    df['has_any_loan'] = ((housing == 'yes') | (loan == 'yes')).astype(np.int8)
    df['is_default'] = (default == 'yes').astype(np.int8)
    df['was_contacted_before'] = (df['pdays'] != -1).astype(np.int8)
    df['is_cellular'] = (contact == 'cellular').astype(np.int8)
    
    # Balance-specific features
    df['balance_signed_log1p'] = np.sign(df['balance']) * np.log1p(df['balance'].abs())
    df['balance_negative'] = (df['balance'] < 0).astype(np.int8)
    df['balance_nonpositive'] = (df['balance'] <= 0).astype(np.int8)
    
    # Bucket features (key to exp012 success)
    pdays_source = df['pdays'].replace(-1, 999)
    
    df['age_bucket'] = pd.cut(
        df['age'], bins=[0, 25, 35, 45, 55, 65, 120],
        labels=['<=25', '26-35', '36-45', '46-55', '56-65', '65+']
    ).astype(str)
    
    df['campaign_bucket'] = pd.cut(
        df['campaign'], bins=[-1, 1, 2, 4, 9, np.inf],
        labels=['1', '2', '3-4', '5-9', '10+']
    ).astype(str)
    
    df['previous_bucket'] = pd.cut(
        df['previous'], bins=[-1, 0, 1, 3, np.inf],
        labels=['0', '1', '2-3', '4+']
    ).astype(str)
    
    df['pdays_bucket'] = pd.cut(
        pdays_source, bins=[-1, 7, 30, 90, 365, np.inf],
        labels=['<=1w', '8-30d', '31-90d', '91-365d', '365d+']
    ).astype(str)
    df.loc[df['pdays'] == -1, 'pdays_bucket'] = 'never'
    
    df['duration_bucket'] = pd.cut(
        df['duration'], bins=[-1, 60, 120, 240, 480, np.inf],
        labels=['<=1m', '1-2m', '2-4m', '4-8m', '8m+']
    ).astype(str)
    
    df['day_bucket'] = pd.cut(
        df['day'], bins=[0, 10, 20, 31],
        labels=['early', 'mid', 'late'], include_lowest=True
    ).astype(str)
    
    # Categorical crosses
    df['job_education'] = job + '__' + education
    df['job_marital'] = job + '__' + marital
    df['contact_month'] = contact + '__' + month
    df['poutcome_month'] = poutcome + '__' + month
    df['loan_default'] = loan + '__' + default
    df['contact_day_bucket'] = contact + '__' + df['day_bucket']
    df['month_day_bucket'] = month + '__' + df['day_bucket']
    
    # History state (critical feature)
    df['history_state'] = np.where(
        df['previous'] > 0,
        poutcome + '__seen',
        'no_previous'
    )
    
    # Define all categorical columns
    categorical_columns = BASE_CATEGORICAL + [
        'age_bucket', 'campaign_bucket', 'previous_bucket', 'pdays_bucket',
        'duration_bucket', 'day_bucket', 'job_education', 'job_marital',
        'contact_month', 'poutcome_month', 'loan_default', 'contact_day_bucket',
        'month_day_bucket', 'history_state'
    ]
    
    # Ensure all categoricals are strings
    for col in categorical_columns:
        df[col] = df[col].fillna('missing').astype(str)
    
    return df, categorical_columns


# ============================================================================
# CATBOOST TRAINING
# ============================================================================

def train_catboost(x_train, x_test, y, cat_cols, params, seeds, use_class_weight=True):
    """
    Train CatBoost with multiple seeds and CV folds.
    Uses class weighting for imbalanced data (like exp012).
    """
    from catboost import CatBoostClassifier
    
    oof_predictions = np.zeros(len(x_train), dtype=np.float64)
    test_predictions = np.zeros(len(x_test), dtype=np.float64)
    
    print(f'\n[CatBoost] Training with {len(seeds)} seeds, {FOLDS} folds')
    
    for seed in seeds:
        seed_oof = np.zeros(len(x_train), dtype=np.float64)
        seed_test = np.zeros(len(x_test), dtype=np.float64)
        
        kfold = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        
        for fold_idx, (train_idx, valid_idx) in enumerate(kfold.split(x_train, y), 1):
            train_x = x_train.iloc[train_idx].reset_index(drop=True)
            train_y = y.iloc[train_idx].reset_index(drop=True)
            valid_x = x_train.iloc[valid_idx].reset_index(drop=True)
            valid_y = y.iloc[valid_idx].reset_index(drop=True)
            
            fold_params = dict(params)
            if use_class_weight:
                fold_params['auto_class_weights'] = 'Balanced'
            
            model = CatBoostClassifier(**fold_params, random_seed=seed)
            model.fit(
                train_x, train_y,
                eval_set=(valid_x, valid_y),
                cat_features=cat_cols,
                use_best_model=True,
                early_stopping_rounds=250,
                verbose=False
            )
            
            seed_oof[valid_idx] = model.predict_proba(valid_x)[:, 1]
            seed_test += model.predict_proba(x_test)[:, 1] / FOLDS
        
        seed_auc = roc_auc_score(y, seed_oof)
        print(f'  Seed {seed}: AUC = {seed_auc:.6f}')
        
        oof_predictions += seed_oof / len(seeds)
        test_predictions += seed_test / len(seeds)
    
    final_auc = roc_auc_score(y, oof_predictions)
    print(f'[CatBoost] Final OOF AUC: {final_auc:.6f}')
    
    return oof_predictions, test_predictions


# ============================================================================
# ATTENTION NEURAL NETWORK (from exp026)
# ============================================================================

def train_attention_nn(x_train, x_test, y, cat_cols, params, seeds):
    """
    Train Attention-based neural network for diversity.
    Uses PyTorch with mixed precision on GPU.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n[AttentionNN] Training on {device} with {len(seeds)} seeds')
    
    # Separate numeric and categorical columns
    num_cols = [c for c in x_train.columns if c not in cat_cols]
    
    # Encode categoricals
    cat_encodings = {}
    cat_dims = []
    x_cat_train = np.zeros((len(x_train), len(cat_cols)), dtype=np.int64)
    x_cat_test = np.zeros((len(x_test), len(cat_cols)), dtype=np.int64)
    
    for idx, col in enumerate(cat_cols):
        combined = pd.concat([
            x_train[col].fillna('missing').astype(str),
            x_test[col].fillna('missing').astype(str)
        ], ignore_index=True)
        codes, _ = pd.factorize(combined, sort=True)
        x_cat_train[:, idx] = codes[:len(x_train)]
        x_cat_test[:, idx] = codes[len(x_train):]
        cat_dims.append(int(codes.max() + 1))
    
    # Scale numeric features
    if num_cols:
        scaler = StandardScaler()
        x_num_train = scaler.fit_transform(
            x_train[num_cols].fillna(x_train[num_cols].median()).astype(np.float32)
        )
        x_num_test = scaler.transform(
            x_test[num_cols].fillna(x_train[num_cols].median()).astype(np.float32)
        )
    else:
        x_num_train = np.zeros((len(x_train), 0), dtype=np.float32)
        x_num_test = np.zeros((len(x_test), 0), dtype=np.float32)
    
    # Define model architecture
    class TransformerBlock(nn.Module):
        def __init__(self, dim, heads, dropout, ff_mult=2):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
            self.norm2 = nn.LayerNorm(dim)
            self.ff = nn.Sequential(
                nn.Linear(dim, dim * ff_mult),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * ff_mult, dim),
                nn.Dropout(dropout)
            )
        
        def forward(self, x):
            attn_in = self.norm1(x)
            attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
            x = x + attn_out
            return x + self.ff(self.norm2(x))
    
    class AttentionNet(nn.Module):
        def __init__(self, num_numeric, cat_dims, emb_dim, hidden_dim, n_layers, heads, dropout):
            super().__init__()
            self.embeddings = nn.ModuleList([nn.Embedding(dim, emb_dim) for dim in cat_dims])
            self.num_proj = nn.Linear(num_numeric, hidden_dim) if num_numeric > 0 else None
            self.cat_proj = nn.Linear(emb_dim, hidden_dim) if cat_dims else None
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            self.blocks = nn.ModuleList([
                TransformerBlock(hidden_dim, heads, dropout) for _ in range(n_layers)
            ])
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        def forward(self, x_num, x_cat):
            tokens = []
            
            # Numeric token
            if self.num_proj is not None and x_num.shape[1] > 0:
                tokens.append(self.num_proj(x_num).unsqueeze(1))
            
            # Categorical tokens
            if self.embeddings:
                for i, emb in enumerate(self.embeddings):
                    cat_emb = emb(x_cat[:, i])
                    tokens.append(self.cat_proj(cat_emb).unsqueeze(1))
            
            # Add CLS token
            batch_size = x_num.shape[0]
            cls = self.cls_token.expand(batch_size, -1, -1)
            tokens.insert(0, cls)
            
            x = torch.cat(tokens, dim=1)
            
            for block in self.blocks:
                x = block(x)
            
            return torch.sigmoid(self.head(x[:, 0]))
    
    oof_predictions = np.zeros(len(x_train), dtype=np.float64)
    test_predictions = np.zeros(len(x_test), dtype=np.float64)
    
    y_np = y.values.astype(np.float32)
    
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        seed_oof = np.zeros(len(x_train), dtype=np.float64)
        seed_test = np.zeros(len(x_test), dtype=np.float64)
        
        kfold = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        
        for fold_idx, (train_idx, valid_idx) in enumerate(kfold.split(x_train, y), 1):
            # Prepare data
            tr_num = torch.FloatTensor(x_num_train[train_idx]).to(device)
            tr_cat = torch.LongTensor(x_cat_train[train_idx]).to(device)
            tr_y = torch.FloatTensor(y_np[train_idx]).unsqueeze(1).to(device)
            
            va_num = torch.FloatTensor(x_num_train[valid_idx]).to(device)
            va_cat = torch.LongTensor(x_cat_train[valid_idx]).to(device)
            va_y = torch.FloatTensor(y_np[valid_idx]).unsqueeze(1).to(device)
            
            te_num = torch.FloatTensor(x_num_test).to(device)
            te_cat = torch.LongTensor(x_cat_test).to(device)
            
            # Create model
            model = AttentionNet(
                num_numeric=x_num_train.shape[1],
                cat_dims=cat_dims,
                emb_dim=params['emb_dim'],
                hidden_dim=params['hidden_dim'],
                n_layers=params['n_layers'],
                heads=params['heads'],
                dropout=params['dropout']
            ).to(device)
            
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=params['lr'],
                weight_decay=params['weight_decay']
            )
            criterion = nn.BCELoss()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=8, factor=0.5
            )
            
            train_dataset = TensorDataset(tr_num, tr_cat, tr_y)
            train_loader = DataLoader(
                train_dataset, batch_size=params['batch_size'], shuffle=True
            )
            
            best_auc = 0
            best_va_pred = None
            best_te_pred = None
            patience_counter = 0
            
            use_amp = device.type == 'cuda'
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
            
            for epoch in range(params['epochs']):
                model.train()
                for batch_num, batch_cat, batch_y in train_loader:
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        output = model(batch_num, batch_cat)
                        loss = criterion(output, batch_y)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                
                model.eval()
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        va_pred = model(va_num, va_cat).cpu().numpy().flatten()
                    
                    auc = roc_auc_score(y.iloc[valid_idx], va_pred)
                    scheduler.step(auc)
                    
                    if auc > best_auc:
                        best_auc = auc
                        best_va_pred = va_pred.copy()
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            best_te_pred = model(te_num, te_cat).cpu().numpy().flatten()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= params['patience']:
                        break
            
            seed_oof[valid_idx] = best_va_pred
            seed_test += best_te_pred / FOLDS
        
        seed_auc = roc_auc_score(y, seed_oof)
        print(f'  Seed {seed}: AUC = {seed_auc:.6f}')
        
        oof_predictions += seed_oof / len(seeds)
        test_predictions += seed_test / len(seeds)
    
    final_auc = roc_auc_score(y, oof_predictions)
    print(f'[AttentionNN] Final OOF AUC: {final_auc:.6f}')
    
    return oof_predictions, test_predictions


# ============================================================================
# BLENDING
# ============================================================================

def find_optimal_blend(oof_dict, y, step=0.01):
    """
    Find optimal blend weights using grid search on OOF predictions.
    Uses rank normalization for robustness.
    """
    names = list(oof_dict.keys())
    
    if len(names) == 2:
        best_auc = -1
        best_weight = 0.5
        
        for w in np.arange(0, 1.0 + step, step):
            blend = w * rank_normalize(oof_dict[names[0]]) + (1 - w) * rank_normalize(oof_dict[names[1]])
            auc = roc_auc_score(y, blend)
            if auc > best_auc:
                best_auc = auc
                best_weight = w
        
        return {names[0]: best_weight, names[1]: 1 - best_weight}, best_auc
    
    # For more models, use uniform weights as fallback
    weight = 1.0 / len(names)
    weights = {name: weight for name in names}
    blend = sum(weights[name] * rank_normalize(oof_dict[name]) for name in names)
    return weights, roc_auc_score(y, blend)


def apply_blend(pred_dict, weights):
    """Apply blend weights to predictions using rank normalization"""
    result = np.zeros(len(next(iter(pred_dict.values()))), dtype=np.float64)
    for name, preds in pred_dict.items():
        result += weights.get(name, 0.0) * rank_normalize(preds)
    return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print('=' * 70)
    print('WINNING SOLUTION - FiiCode 2026 AI Competition')
    print('=' * 70)
    
    # Load data
    data_dir = locate_data()
    train = pd.read_csv(data_dir / 'train.csv')
    test = pd.read_csv(data_dir / 'test.csv')
    
    y = train[TARGET].astype(int)
    train_features = train.drop(columns=[TARGET])
    
    print(f'\nDataset: {len(train)} train, {len(test)} test')
    print(f'Target distribution: {y.mean():.4f} positive')
    
    # Build features
    print('\n[Features] Building blend_bucket features...')
    x_train, cat_cols = build_blend_bucket_features(train_features)
    x_test, _ = build_blend_bucket_features(test)
    print(f'Created {len(x_train.columns)} features, {len(cat_cols)} categorical')
    
    # Store predictions
    oof_preds = {}
    test_preds = {}
    
    # Train CatBoost
    cb_oof, cb_test = train_catboost(
        x_train, x_test, y, cat_cols, CATBOOST_PARAMS, CATBOOST_SEEDS
    )
    oof_preds['catboost'] = cb_oof
    test_preds['catboost'] = cb_test
    
    # Train Attention NN (if PyTorch available)
    try:
        import torch
        nn_oof, nn_test = train_attention_nn(
            x_train, x_test, y, cat_cols, NN_PARAMS, NN_SEEDS
        )
        oof_preds['attention'] = nn_oof
        test_preds['attention'] = nn_test
    except ImportError:
        print('\n[Warning] PyTorch not available, skipping NN training')
    
    # Find optimal blend
    print('\n' + '=' * 70)
    print('BLENDING')
    print('=' * 70)
    
    if len(oof_preds) > 1:
        # Find optimal weights
        weights, blend_auc = find_optimal_blend(oof_preds, y)
        print(f'\nOptimal weights: {weights}')
        print(f'Optimal blend OOF AUC: {blend_auc:.6f}')
        
        # Also try fixed 80/20 (proven on LB)
        fixed_weights = {'catboost': 0.80, 'attention': 0.20}
        fixed_blend = apply_blend(oof_preds, fixed_weights)
        fixed_auc = roc_auc_score(y, fixed_blend)
        print(f'Fixed 80/20 blend OOF AUC: {fixed_auc:.6f}')
        
        # Use whichever is better
        if fixed_auc >= blend_auc:
            weights = fixed_weights
            blend_auc = fixed_auc
            print('Using fixed 80/20 weights (proven on LB)')
        
        final_test = apply_blend(test_preds, weights)
    else:
        # Single model
        name = list(test_preds.keys())[0]
        final_test = rank_normalize(test_preds[name])
        blend_auc = roc_auc_score(y, oof_preds[name])
    
    # Generate submission
    print('\n' + '=' * 70)
    print('GENERATING SUBMISSION')
    print('=' * 70)
    
    submission = pd.DataFrame({
        ID_COL: test[ID_COL],
        TARGET: np.clip(final_test, 1e-6, 1 - 1e-6)
    })
    submission_path = submission_output_path("submission.csv")
    submission.to_csv(submission_path, index=False)
    
    print(f'\nFinal OOF AUC: {blend_auc:.6f}')
    print(f'Saved: {submission_path}')
    print(submission.head(10))
    
    # Also save individual model submissions
    for name, preds in test_preds.items():
        sub = pd.DataFrame({
            ID_COL: test[ID_COL],
            TARGET: np.clip(rank_normalize(preds), 1e-6, 1 - 1e-6)
        })
        sub_path = submission_output_path(f"submission_{name}.csv")
        sub.to_csv(sub_path, index=False)
        print(f'Saved: {sub_path}')


if __name__ == '__main__':
    main()
