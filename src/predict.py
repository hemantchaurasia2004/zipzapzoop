"""Generate submission.csv from trained models."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import Booster

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODEL_DIR, SUBMISSION_PATH, TEST_PATH  # noqa: E402
from src.data_loader import load_test  # noqa: E402
from src.features import FeatureEngineer  # noqa: E402


def _prepare_x(df: pd.DataFrame, feature_cols: list[str], cat_cols: list[str], for_lightgbm: bool = False) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype(str).fillna("missing")
            if for_lightgbm:
                X[c] = X[c].astype("category")
    return X


def predict(test_df: pd.DataFrame, model_dir: Path | None = None) -> pd.DataFrame:
    model_dir = Path(model_dir or MODEL_DIR)
    meta = json.loads((model_dir / "train_meta.json").read_text())
    engineer = FeatureEngineer.load(model_dir / "feature_engineer")

    test_feat = engineer.transform(test_df)
    feature_cols = meta["feature_cols"]
    cat_cols = meta["cat_cols"]
    X = _prepare_x(test_feat, feature_cols, cat_cols)
    X_lgb = _prepare_x(test_feat, feature_cols, cat_cols, for_lightgbm=True)

    cat_model = CatBoostRegressor()
    cat_model.load_model(str(model_dir / "catboost.cbm"))
    cat_idx = [i for i, c in enumerate(feature_cols) if c in cat_cols]
    pred_cb = cat_model.predict(Pool(X, cat_features=cat_idx))

    lgb_model = Booster(model_file=str(model_dir / "lightgbm.txt"))
    pred_lgb = lgb_model.predict(X_lgb)

    w = meta["blend_weight_catboost"]
    pred_log = w * pred_cb + (1 - w) * pred_lgb
    if meta.get("use_log_target", True):
        pred = np.expm1(pred_log)
    else:
        pred = pred_log
    pred = np.clip(pred, 1e-7, 1.0)

    return pd.DataFrame({"Index": test_df["Index"].values, "demand": pred})


def validate_submission(sub: pd.DataFrame, test_df: pd.DataFrame) -> None:
    assert len(sub) == 41_778, f"Expected 41778 rows, got {len(sub)}"
    assert list(sub.columns) == ["Index", "demand"], f"Bad columns: {sub.columns.tolist()}"
    assert sub["demand"].notna().all(), "NaN predictions found"
    assert (sub["demand"] > 0).all(), "Non-positive predictions found"
    assert sub["Index"].tolist() == test_df["Index"].tolist(), "Index order mismatch"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output", type=Path, default=SUBMISSION_PATH)
    args = parser.parse_args()

    test_df = load_test()
    sub = predict(test_df, args.model_dir)
    validate_submission(sub, test_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.output, index=False)
    print(f"Submission saved: {args.output} ({len(sub)} rows)")
    print(sub["demand"].describe())


if __name__ == "__main__":
    main()
