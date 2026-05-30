"""
High-accuracy traffic demand pipeline (v2).

Combines 2ps.py strengths (5-fold CV, raw R² target, timestamp categorical)
with lag features, OOF target encodings, and day-48 blend optimization.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import geohash
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, OUTPUT_DIR, RANDOM_SEED  # noqa: E402

TARGET = "demand"
N_FOLDS = 5
CAT_COLS = [
    "geohash",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather",
    "timestamp",
    "geohash_prefix4",
]
DROP_COLS = {"Index", TARGET}


def _decode(g: str) -> tuple[float, float]:
    try:
        return geohash.decode(g)
    except Exception:
        return 0.0, 0.0


def _parse_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.str.strip()
    out["timestamp"] = out["timestamp"].astype(str)
    out["hour"] = out["timestamp"].apply(lambda x: int(x.split(":")[0]))
    out["minute"] = out["timestamp"].apply(lambda x: int(x.split(":")[1]))
    out["slot_idx"] = out["hour"] * 4 + out["minute"] // 15
    out["time_in_minutes"] = out["hour"] * 60 + out["minute"]
    return out


def _impute(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr, te = train.copy(), test.copy()
    for col in tr.columns:
        if col == TARGET:
            continue
        if tr[col].dtype == "object" or col in ("RoadType", "Weather", "LargeVehicles", "Landmarks"):
            tr[col] = tr[col].replace("", np.nan).fillna("Unknown")
            if col in te.columns:
                te[col] = te[col].replace("", np.nan).fillna("Unknown")
        else:
            med = tr[col].median()
            tr[col] = tr[col].fillna(med)
            if col in te.columns:
                te[col] = te[col].fillna(med)
    return tr, te


def _base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lats, lons = zip(*[_decode(g) for g in out["geohash"]])
    out["latitude"] = lats
    out["longitude"] = lons
    out["geohash_prefix4"] = out["geohash"].str[:4]

    out["is_morning_rush"] = out["hour"].isin([7, 8, 9]).astype(int)
    out["is_evening_rush"] = out["hour"].isin([17, 18, 19]).astype(int)
    out["is_night"] = out["hour"].isin([0, 1, 2, 3, 4]).astype(int)
    out["is_midday"] = out["hour"].between(10, 14).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["minute_sin"] = np.sin(2 * np.pi * out["minute"] / 60)
    out["minute_cos"] = np.cos(2 * np.pi * out["minute"] / 60)
    out["slot_sin"] = np.sin(2 * np.pi * out["slot_idx"] / 96)
    out["slot_cos"] = np.cos(2 * np.pi * out["slot_idx"] / 96)
    out["day_sin"] = np.sin(2 * np.pi * out["day"] / 7)
    out["day_cos"] = np.cos(2 * np.pi * out["day"] / 7)

    weather_map = {"Sunny": 0, "Foggy": 1, "Rainy": 2, "Snowy": 3, "Unknown": 0}
    out["weather_encoded"] = out["Weather"].map(weather_map).fillna(0).astype(int)

    out["lat_lon_interaction"] = out["latitude"] * out["longitude"]
    out["temp_lane_interaction"] = out["Temperature"] * out["NumberofLanes"]
    out["hour_lane_interaction"] = out["hour"] * out["NumberofLanes"]
    out["weather_temp"] = out["weather_encoded"] * out["Temperature"]
    out["large_vehicles_allowed"] = (out["LargeVehicles"] == "Allowed").astype(int)
    out["has_landmarks"] = (out["Landmarks"] == "Yes").astype(int)
    return out


def _smooth_encoding(series: pd.Series, target: pd.Series, m: float = 20.0) -> pd.Series:
    g = target.groupby(series).agg(["mean", "count"])
    global_mean = target.mean()
    enc = (g["count"] * g["mean"] + m * global_mean) / (g["count"] + m)
    return series.map(enc).fillna(global_mean)


def add_history_features(
    df: pd.DataFrame,
    history: pd.DataFrame,
    global_mean: float,
) -> pd.DataFrame:
    """Add demand history features from `history` (typically train fold)."""
    out = df.copy()
    h = history

    d48 = h[h["day"] == 48][["geohash", "timestamp", TARGET]].rename(columns={TARGET: "demand_d48_slot"})
    out = out.merge(d48, on=["geohash", "timestamp"], how="left")

    gh_ts = h.groupby(["geohash", "timestamp"])[TARGET].mean().rename("te_geohash_ts")
    gh_h = h.groupby(["geohash", "hour"])[TARGET].mean().rename("te_geohash_hour")
    gh = h.groupby("geohash")[TARGET].mean().rename("te_geohash")
    ts = h.groupby("timestamp")[TARGET].mean().rename("te_timestamp")
    rt_h = h.groupby(["RoadType", "hour"])[TARGET].mean().rename("te_road_hour")

    out = out.merge(gh_ts, on=["geohash", "timestamp"], how="left")
    out = out.merge(gh_h, on=["geohash", "hour"], how="left")
    out = out.merge(gh, on="geohash", how="left")
    out = out.merge(ts, on="timestamp", how="left")
    out = out.merge(rt_h, on=["RoadType", "hour"], how="left")

    for col in ["te_geohash_ts", "te_geohash_hour", "te_geohash", "te_timestamp", "te_road_hour"]:
        out[col] = out[col].fillna(global_mean)
    out["demand_d48_slot"] = out["demand_d48_slot"].fillna(out["te_geohash_ts"])
    return out


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DROP_COLS]


def _label_encode_cats(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr, va, te = X_train.copy(), X_valid.copy(), X_test.copy()
    for col in cat_cols:
        if col not in tr.columns:
            continue
        combined = pd.concat([tr[col], va[col], te[col]]).astype(str)
        mapping = {k: v for v, k in enumerate(combined.unique())}
        tr[col] = tr[col].astype(str).map(mapping)
        va[col] = va[col].astype(str).map(mapping).fillna(-1).astype(int)
        te[col] = te[col].astype(str).map(mapping).fillna(-1).astype(int)
    return tr, va, te


def _prepare_catboost(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    out = X.copy()
    for c in cat_cols:
        if c in out.columns:
            out[c] = out[c].astype(str)
    return out


def optimize_d48_blend(oof: np.ndarray, d48: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    best_r2, best_w = -1.0, 0.0
    for w in np.arange(0.0, 1.01, 0.05):
        pred = np.clip(w * d48 + (1 - w) * oof, 0, None)
        r2 = r2_score(y, pred)
        if r2 > best_r2:
            best_r2, best_w = r2, w
    return best_w, best_r2


def run_pipeline(
    train_path: Path | None = None,
    test_path: Path | None = None,
    output_path: Path | None = None,
    n_folds: int = N_FOLDS,
) -> dict:
    train_path = train_path or DATA_DIR / "train.csv"
    test_path = test_path or DATA_DIR / "test.csv"
    output_path = output_path or OUTPUT_DIR / "submission_v2.csv"

    train = _parse_time(pd.read_csv(train_path))
    test = _parse_time(pd.read_csv(test_path))
    train, test = _impute(train, test)

    train = _base_features(train)
    test = _base_features(test)

    y = train[TARGET].values
    global_mean = float(y.mean())

    # Full-history features for test (all train rows — no label leakage on test)
    test_full = add_history_features(test, train, global_mean)

    feature_cols = _feature_columns(train)
    # History columns added at fold time for train; ensure test has them
    hist_cols = [
        "demand_d48_slot",
        "te_geohash_ts",
        "te_geohash_hour",
        "te_geohash",
        "te_timestamp",
        "te_road_hour",
    ]
    for c in hist_cols:
        if c not in feature_cols:
            feature_cols.append(c)

    X_test_base = test_full[feature_cols]
    X_test_cb = _prepare_catboost(X_test_base, CAT_COLS)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(train))
    oof_d48 = np.zeros(len(train))
    test_cat = np.zeros(len(test))
    test_lgb = np.zeros(len(test))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train)):
        print(f"\n{'=' * 50}\nFold {fold + 1}/{n_folds}\n{'=' * 50}")

        tr_df = add_history_features(train.iloc[tr_idx], train.iloc[tr_idx], global_mean)
        va_df = add_history_features(train.iloc[va_idx], train.iloc[tr_idx], global_mean)

        X_tr = tr_df[feature_cols]
        X_va = va_df[feature_cols]
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        X_tr_cb = _prepare_catboost(X_tr, CAT_COLS)
        X_va_cb = _prepare_catboost(X_va, CAT_COLS)

        cat_model = CatBoostRegressor(
            iterations=4000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=5,
            loss_function="RMSE",
            eval_metric="R2",
            random_seed=RANDOM_SEED + fold,
            verbose=300,
        )
        cat_model.fit(
            X_tr_cb,
            y_tr,
            cat_features=[c for c in CAT_COLS if c in X_tr_cb.columns],
            eval_set=(X_va_cb, y_va),
            early_stopping_rounds=250,
            use_best_model=True,
        )
        va_cat = cat_model.predict(X_va_cb)
        test_cat += cat_model.predict(X_test_cb) / n_folds

        X_tr_lgb, X_va_lgb, X_te_lgb = _label_encode_cats(
            X_tr, X_va, X_test_base, CAT_COLS
        )
        lgb_model = LGBMRegressor(
            objective="regression",
            n_estimators=4000,
            learning_rate=0.02,
            num_leaves=128,
            max_depth=10,
            min_child_samples=15,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=0.5,
            random_state=RANDOM_SEED + fold,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(
            X_tr_lgb,
            y_tr,
            eval_set=[(X_va_lgb, y_va)],
            eval_metric="l2",
            callbacks=[early_stopping(250, verbose=False)],
        )
        va_lgb = lgb_model.predict(X_va_lgb)
        test_lgb += lgb_model.predict(X_te_lgb) / n_folds

        va_ens = 0.7 * va_cat + 0.3 * va_lgb
        oof[va_idx] = va_ens
        oof_d48[va_idx] = va_df["demand_d48_slot"].values

        fold_r2 = r2_score(y_va, va_ens)
        print(f"Fold ensemble R²: {fold_r2:.6f}")

    oof_r2 = r2_score(y, oof)
    print(f"\nOOF ensemble R²: {oof_r2:.6f}")

    d48_w, blend_r2 = optimize_d48_blend(oof, oof_d48, y)
    print(f"Optimal day-48 blend weight: {d48_w:.2f} (OOF R²: {blend_r2:.6f})")

    test_ens = 0.7 * test_cat + 0.3 * test_lgb
    test_d48 = test_full["demand_d48_slot"].values
    final_preds = np.clip(d48_w * test_d48 + (1 - d48_w) * test_ens, 0, None)

    submission = pd.DataFrame({"Index": test["Index"], "demand": final_preds})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    meta = {
        "oof_r2": oof_r2,
        "blend_r2_with_d48": blend_r2,
        "d48_weight": d48_w,
        "n_folds": n_folds,
        "feature_count": len(feature_cols),
    }
    (OUTPUT_DIR / "submission_v2_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nSaved {output_path} ({len(submission)} rows)")
    print(submission["demand"].describe())
    return meta


def main():
    parser = argparse.ArgumentParser(description="Train v2 high-accuracy pipeline")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "submission_v2.csv")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    args = parser.parse_args()
    run_pipeline(output_path=args.output, n_folds=args.folds)


if __name__ == "__main__":
    main()
