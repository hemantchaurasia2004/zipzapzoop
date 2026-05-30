"""Train CatBoost + LightGBM ensemble with day-49 holdout validation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor, early_stopping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    BLEND_WEIGHT_CATBOOST,
    CATBOOST_PARAMS,
    CAT_FEATURES,
    LIGHTGBM_PARAMS,
    MODEL_DIR,
    OPTUNA_TRIALS,
    RANDOM_SEED,
)
from src.data_loader import load_train, split_train_holdout  # noqa: E402
from src.features import FeatureEngineer  # noqa: E402
from src.metrics import regression_metrics, scaled_r2  # noqa: E402


def _prepare_xy(df: pd.DataFrame, feature_cols: list[str], cat_cols: list[str], for_lightgbm: bool = False):
    X = df[feature_cols].copy()
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype(str).fillna("missing")
            if for_lightgbm:
                X[c] = X[c].astype("category")
    y = df["demand"].values if "demand" in df.columns else None
    cat_idx = [i for i, c in enumerate(feature_cols) if c in cat_cols]
    return X, y, cat_idx


def train_models(
    train_feat: pd.DataFrame,
    val_feat: pd.DataFrame,
    feature_cols: list[str],
    use_log_target: bool = True,
    tune: bool = False,
) -> dict:
    cat_cols = [c for c in CAT_FEATURES if c in feature_cols]
    X_train, y_train, cat_idx = _prepare_xy(train_feat, feature_cols, cat_cols)
    X_val, y_val, _ = _prepare_xy(val_feat, feature_cols, cat_cols)
    X_train_lgb, _, _ = _prepare_xy(train_feat, feature_cols, cat_cols, for_lightgbm=True)
    X_val_lgb, _, _ = _prepare_xy(val_feat, feature_cols, cat_cols, for_lightgbm=True)

    if use_log_target:
        y_train_fit = np.log1p(y_train)
        y_val_fit = np.log1p(y_val)
    else:
        y_train_fit = y_train
        y_val_fit = y_val

    cb_params = dict(CATBOOST_PARAMS)
    lgb_params = dict(LIGHTGBM_PARAMS)

    if tune:
        cb_params, lgb_params = _optuna_tune(
            X_train, X_train_lgb, y_train_fit, X_val, X_val_lgb, y_val_fit, cat_idx, cat_cols
        )

    train_pool = Pool(X_train, y_train_fit, cat_features=cat_idx)
    val_pool = Pool(X_val, y_val_fit, cat_features=cat_idx)

    cat_model = CatBoostRegressor(**cb_params)
    cat_model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    lgb_model = LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train_lgb,
        y_train_fit,
        eval_set=[(X_val_lgb, y_val_fit)],
        callbacks=[early_stopping(150, verbose=False)],
        categorical_feature=cat_cols if cat_cols else "auto",
    )

    w = BLEND_WEIGHT_CATBOOST
    pred_log = w * cat_model.predict(X_val) + (1 - w) * lgb_model.predict(X_val_lgb)
    pred = np.expm1(pred_log) if use_log_target else pred_log
    pred = np.clip(pred, 1e-7, 1.0)

    metrics = regression_metrics(y_val, pred)
    return {
        "catboost": cat_model,
        "lightgbm": lgb_model,
        "metrics": metrics,
        "blend_weight_catboost": w,
        "use_log_target": use_log_target,
        "cb_params": cb_params,
        "lgb_params": {k: v for k, v in lgb_params.items() if k != "verbose"},
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
    }


def _optuna_tune(X_train, X_train_lgb, y_train, X_val, X_val_lgb, y_val, cat_idx, cat_cols):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        cb = CatBoostRegressor(
            loss_function="RMSE",
            iterations=1500,
            learning_rate=trial.suggest_float("cb_lr", 0.02, 0.12, log=True),
            depth=trial.suggest_int("cb_depth", 6, 10),
            l2_leaf_reg=trial.suggest_float("cb_l2", 1.0, 12.0),
            random_seed=RANDOM_SEED,
            verbose=0,
            early_stopping_rounds=100,
        )
        train_pool = Pool(X_train, y_train, cat_features=cat_idx)
        val_pool = Pool(X_val, y_val, cat_features=cat_idx)
        cb.fit(train_pool, eval_set=val_pool, use_best_model=True)
        pred_cb = cb.predict(X_val)

        lgb = LGBMRegressor(
            objective="regression",
            n_estimators=1500,
            learning_rate=trial.suggest_float("lgb_lr", 0.02, 0.12, log=True),
            num_leaves=trial.suggest_int("lgb_leaves", 31, 127),
            max_depth=trial.suggest_int("lgb_depth", 6, 12),
            min_child_samples=trial.suggest_int("lgb_min_child", 10, 50),
            random_state=RANDOM_SEED,
            verbose=-1,
        )
        lgb.fit(
            X_train_lgb,
            y_train,
            eval_set=[(X_val_lgb, y_val)],
            callbacks=[early_stopping(100, verbose=False)],
            categorical_feature=cat_cols if cat_cols else "auto",
        )
        pred = 0.55 * pred_cb + 0.45 * lgb.predict(X_val_lgb)
        y_true = np.expm1(y_val)
        pred = np.clip(np.expm1(pred), 1e-7, 1.0)
        return scaled_r2(y_true, pred)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
    bp = study.best_params

    cb_params = {
        **CATBOOST_PARAMS,
        "learning_rate": bp["cb_lr"],
        "depth": bp["cb_depth"],
        "l2_leaf_reg": bp["cb_l2"],
    }
    lgb_params = {
        **LIGHTGBM_PARAMS,
        "learning_rate": bp["lgb_lr"],
        "num_leaves": bp["lgb_leaves"],
        "max_depth": bp["lgb_depth"],
        "min_child_samples": bp["lgb_min_child"],
    }
    print(f"Optuna best scaled R²: {study.best_value:.2f}")
    return cb_params, lgb_params


def fit_full_and_save(
    train_df: pd.DataFrame,
    artifacts: dict,
    engineer: FeatureEngineer,
) -> None:
    """Retrain on full training data and persist models."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    full_feat = engineer.transform(train_df)
    feature_cols = artifacts["feature_cols"]
    cat_cols = artifacts["cat_cols"]
    X, y, cat_idx = _prepare_xy(full_feat, feature_cols, cat_cols)
    X_lgb, _, _ = _prepare_xy(full_feat, feature_cols, cat_cols, for_lightgbm=True)
    use_log = artifacts["use_log_target"]
    y_fit = np.log1p(y) if use_log else y

    cb = CatBoostRegressor(**artifacts["cb_params"])
    cb.fit(Pool(X, y_fit, cat_features=cat_idx))
    cb.save_model(str(MODEL_DIR / "catboost.cbm"))

    lgb = LGBMRegressor(**artifacts["lgb_params"])
    lgb.fit(X_lgb, y_fit, categorical_feature=cat_cols if cat_cols else "auto")
    lgb.booster_.save_model(str(MODEL_DIR / "lightgbm.txt"))

    meta = {
        "blend_weight_catboost": artifacts["blend_weight_catboost"],
        "use_log_target": use_log,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "cb_params": {k: v for k, v in artifacts["cb_params"].items() if k != "verbose"},
        "lgb_params": artifacts["lgb_params"],
        "holdout_metrics": artifacts["metrics"],
    }
    (MODEL_DIR / "train_meta.json").write_text(json.dumps(meta, indent=2))
    engineer.save(MODEL_DIR / "feature_engineer")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter search")
    parser.add_argument("--no-log-target", action="store_true")
    args = parser.parse_args()

    train_raw = load_train()
    train_split, val_split = split_train_holdout(train_raw)

    engineer = FeatureEngineer()
    engineer.fit(train_split)
    train_feat = engineer.transform(train_split)
    val_feat = engineer.transform(val_split)
    feature_cols = engineer.get_feature_columns()

    print(f"Train rows: {len(train_feat)}, holdout rows: {len(val_feat)}")
    artifacts = train_models(
        train_feat,
        val_feat,
        feature_cols,
        use_log_target=not args.no_log_target,
        tune=args.tune,
    )
    print("Holdout metrics:", artifacts["metrics"])

    engineer_full = FeatureEngineer()
    engineer_full.fit(train_raw)
    fit_full_and_save(train_raw, artifacts, engineer_full)
    print(f"Models saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
