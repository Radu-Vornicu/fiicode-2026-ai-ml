import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from submission_layout import resolve_submission_path, submission_output_path

TARGET = "Subscribed"
ID_COL = "id"

SOURCE_REPO = Path(r"C:\Users\dbxdr_iytiz92\Dropbox\fiicode")
TARGET_REPO = Path(__file__).resolve().parent
ARTIFACT_ROOT = TARGET_REPO / "reference_configs"


def rank_normalize(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)


def validate_submission(frame: pd.DataFrame, *, name: str) -> None:
    if list(frame.columns) != [ID_COL, TARGET]:
        raise ValueError(f"{name} has invalid columns: {list(frame.columns)}")
    if len(frame) != 1000:
        raise ValueError(f"{name} has {len(frame)} rows, expected 1000")
    if frame[TARGET].isna().any():
        raise ValueError(f"{name} contains NaN predictions")
    if not np.isfinite(frame[TARGET]).all():
        raise ValueError(f"{name} contains non-finite predictions")


def load_source_oof(experiment_name: str) -> pd.DataFrame:
    path = SOURCE_REPO / "outputs" / "oof" / experiment_name / "oof_predictions.csv"
    return pd.read_csv(path).sort_values(ID_COL).reset_index(drop=True)


def load_source_submission(experiment_name: str) -> pd.DataFrame:
    path = SOURCE_REPO / "outputs" / "submissions" / experiment_name / "submission.csv"
    return pd.read_csv(path).sort_values(ID_COL).reset_index(drop=True)


def load_target_frame(filename: str) -> pd.DataFrame:
    path = resolve_submission_path(filename, TARGET_REPO)
    return pd.read_csv(path).sort_values(ID_COL).reset_index(drop=True)


def build_exact_source_public_best() -> tuple[pd.DataFrame, float, dict]:
    exp012_oof = load_source_oof("exp012_blend_bucket_features_fixed")
    nnblend_oof = load_source_oof("exp026_gpu_nn_blend")
    source_public_best = load_source_submission("exp_blend_80_20")

    if not exp012_oof[ID_COL].equals(nnblend_oof[ID_COL]):
        raise ValueError("Source OOF ids are misaligned for exp012 and exp026_gpu_nn_blend")

    y = exp012_oof["y_true"].to_numpy(dtype=int)
    oof_blend = 0.80 * rank_normalize(exp012_oof["oof_pred"].to_numpy(dtype=float))
    oof_blend += 0.20 * rank_normalize(nnblend_oof["oof_pred"].to_numpy(dtype=float))
    auc = roc_auc_score(y, oof_blend)

    validate_submission(source_public_best, name="source_public_best")
    return source_public_best, auc, {
        "type": "exact_source_copy",
        "source_submission": str(
            SOURCE_REPO / "outputs" / "submissions" / "exp_blend_80_20" / "submission.csv"
        ),
        "oof_components": [
            "exp012_blend_bucket_features_fixed",
            "exp026_gpu_nn_blend",
        ],
        "weights": {
            "exp012_blend_bucket_features_fixed": 0.80,
            "exp026_gpu_nn_blend": 0.20,
        },
        "notes": "Exact copy of the source repo public-best 80/20 blend submission, scored locally via the underlying OOF blend.",
    }


def build_hybrid_rank_blend() -> tuple[pd.DataFrame, float, dict]:
    nn_oof = load_source_oof("exp026_gpu_nn_blend")
    overnight_oof = load_target_frame("oof_overnight_push.csv")
    bundle_oof = load_target_frame("oof_bundle_catboost.csv")

    overnight_sub = load_target_frame("submission_overnight_pseudo5_best.csv")
    bundle_sub = load_target_frame("submission_bundle_state3way_best.csv")
    nn_sub = load_source_submission("exp026_gpu_nn_blend")

    if not overnight_oof[ID_COL].equals(bundle_oof[ID_COL]):
        raise ValueError("Target OOF ids are misaligned between overnight and bundle assets")
    if not overnight_oof[ID_COL].equals(nn_oof[ID_COL]):
        raise ValueError("Source and target OOF ids are misaligned")

    y = overnight_oof[TARGET].to_numpy(dtype=int)
    oof_blend = 0.50 * rank_normalize(overnight_oof["overnight_pseudo5_best"].to_numpy(dtype=float))
    oof_blend += 0.20 * rank_normalize(nn_oof["oof_pred"].to_numpy(dtype=float))
    oof_blend += 0.30 * rank_normalize(bundle_oof["bundle_state3way_best"].to_numpy(dtype=float))
    auc = roc_auc_score(y, oof_blend)

    for name, frame in [
        ("submission_overnight_pseudo5_best.csv", overnight_sub),
        ("submission_bundle_state3way_best.csv", bundle_sub),
        ("source_exp026_gpu_nn_blend_submission", nn_sub),
    ]:
        validate_submission(frame, name=name)

    if not overnight_sub[ID_COL].equals(bundle_sub[ID_COL]):
        raise ValueError("Target submission ids are misaligned between overnight and bundle assets")
    if not overnight_sub[ID_COL].equals(nn_sub[ID_COL]):
        raise ValueError("Source and target submission ids are misaligned")

    hybrid_pred = 0.50 * rank_normalize(overnight_sub[TARGET].to_numpy(dtype=float))
    hybrid_pred += 0.20 * rank_normalize(nn_sub[TARGET].to_numpy(dtype=float))
    hybrid_pred += 0.30 * rank_normalize(bundle_sub[TARGET].to_numpy(dtype=float))

    hybrid_submission = pd.DataFrame(
        {
            ID_COL: overnight_sub[ID_COL].to_numpy(),
            TARGET: np.clip(hybrid_pred, 1e-6, 1.0 - 1e-6),
        }
    )
    validate_submission(hybrid_submission, name="hybrid_rank_blend")
    return hybrid_submission, auc, {
        "type": "rank_blend",
        "components": {
            "submission_overnight_pseudo5_best.csv": 0.50,
            str(SOURCE_REPO / "outputs" / "submissions" / "exp026_gpu_nn_blend" / "submission.csv"): 0.20,
            "submission_bundle_state3way_best.csv": 0.30,
        },
        "notes": "OOF-optimized 3-way rank blend that adds the source neural branch to Radu's two strongest local ensemble assets.",
    }


def main() -> None:
    ARTIFACT_ROOT.mkdir(exist_ok=True)

    exact_source_submission, exact_auc, exact_meta = build_exact_source_public_best()
    hybrid_submission, hybrid_auc, hybrid_meta = build_hybrid_rank_blend()

    exact_path = submission_output_path("submission_top3_catboost_exact_match.csv", TARGET_REPO)
    hybrid_path = submission_output_path("submission_top3_catboost_hybrid_blend.csv", TARGET_REPO)
    exact_source_submission.to_csv(exact_path, index=False)
    hybrid_submission.to_csv(hybrid_path, index=False)

    summary = {
        "generated_at_repo": str(TARGET_REPO),
        "source_repo": str(SOURCE_REPO),
        "candidates": [
            {
                "name": "top3_catboost_exact_match",
                "path": str(exact_path),
                "local_oof_auc": exact_auc,
                **exact_meta,
            },
            {
                "name": "top3_catboost_hybrid_blend",
                "path": str(hybrid_path),
                "local_oof_auc": hybrid_auc,
                **hybrid_meta,
            },
        ],
        "recommended_submission": str(exact_path),
        "recommended_reason": "This is the exact copy of the source repo public-best branch and is the safest top-3 candidate.",
        "challenger_submission": str(hybrid_path),
        "challenger_reason": "This hybrid improves the target repo's best available local OOF blend by injecting the source neural branch for extra diversity.",
    }

    summary_path = ARTIFACT_ROOT / "solution_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 72)
    print("CatBoost Ensemble Submission Builder")
    print("=" * 72)
    print(f"Exact source copy local OOF AUC: {exact_auc:.6f}")
    print(f"Hybrid rank blend local OOF AUC: {hybrid_auc:.6f}")
    print(f"Wrote: {exact_path}")
    print(f"Wrote: {hybrid_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
