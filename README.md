# FiiCode 2026 - Bank Telemarketing Prediction

## CatBoost Ensemble Solution

This repo contains a CatBoost-based solution for predicting bank telemarketing success.

### Key Scripts

- `catboost_blend_solution.py` - Main ensemble solution
- `catboost_training_utils.py` - Training utilities and configs
- `winning_solution.py` - Full implementation with blend_buckets features

### Run

```bash
python catboost_blend_solution.py
```

### Outputs

- `submission_catboost_blend_catboost_exact_match/submission.csv`
- `submission_catboost_blend_catboost_hybrid_blend/submission.csv`
- `reference_configs/solution_summary.json`

## Submission Layout

Submissions now follow a folder-based layout:

```text
submission_name/
  submission.csv
```

The scripts in this repo were updated to read and write submissions using that
layout.
