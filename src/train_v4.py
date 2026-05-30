"""
Traffic demand pipeline v4 — 3-model stacking with advanced features.
KFold CV approach (local execution). Targets 95+ leaderboard (R² × 100).

Key improvements over v3:
  1. XGBoost as 3rd base model for ensemble diversity
  2. Ridge meta-learner instead of fixed 70/30 blend
  3. Expanded d48 lag features (±1 slot, rolling stats, hour-level)
  4. Multi-granularity geohash target encodings (prefix3/4/5)
  5. Interaction target encodings (RoadType×hour, Weather×hour)
  6. Geohash-level post-processing calibration
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import geohash as gh_lib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, OUTPUT_DIR, RANDOM_SEED  # noqa: E402

TARGET = "demand"
N_FOLDS = 5
SMOOTHING_M = 20

CAT_FEATURES = [
    "geohash", "RoadType", "LargeVehicles", "Landmarks",
    "Weather", "timestamp", "geohash_prefix4",
]

BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
B32_MAP = {c: i for i, c in enumerate(BASE32)}


# ============================================================
# Utilities
# ============================================================
def _decode(g: str):
    try:
        la, lo = gh_lib.decode(g)
        return float(la), float(lo)
    except Exception:
        return 0.0, 0.0


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    train.columns = train.columns.str.strip()
    test.columns = test.columns.str.strip()
    return train, test


def impute(train, test):
    for col in train.columns:
        if col == TARGET:
            continue
        if train[col].dtype == "object":
            train[col] = train[col].replace("", np.nan).fillna("Unknown")
            if col in test.columns:
                test[col] = test[col].replace("", np.nan).fillna("Unknown")
        else:
            med = train[col].median()
            train[col] = train[col].fillna(med)
            if col in test.columns:
                test[col] = test[col].fillna(med)


# ============================================================
# Base feature engineering (no target leakage)
# ============================================================
def create_base_features(df):
    out = df.copy()
    out["timestamp"] = out["timestamp"].astype(str)
    out["hour"] = out["timestamp"].apply(lambda x: int(x.split(":")[0]))
    out["minute"] = out["timestamp"].apply(lambda x: int(x.split(":")[1]))
    out["time_in_minutes"] = out["hour"] * 60 + out["minute"]
    out["slot_idx"] = out["hour"] * 4 + out["minute"] // 15

    # Cyclical
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["minute_sin"] = np.sin(2 * np.pi * out["minute"] / 60)
    out["minute_cos"] = np.cos(2 * np.pi * out["minute"] / 60)
    out["slot_sin"] = np.sin(2 * np.pi * out["slot_idx"] / 96)
    out["slot_cos"] = np.cos(2 * np.pi * out["slot_idx"] / 96)
    out["day_sin"] = np.sin(2 * np.pi * out["day"] / 7)
    out["day_cos"] = np.cos(2 * np.pi * out["day"] / 7)

    # Time-of-day flags
    out["is_morning_rush"] = out["hour"].isin([7, 8, 9]).astype(int)
    out["is_evening_rush"] = out["hour"].isin([17, 18, 19]).astype(int)
    out["is_night"] = out["hour"].isin([0, 1, 2, 3, 4]).astype(int)
    out["is_midday"] = out["hour"].between(10, 14).astype(int)
    out["is_rush"] = (out["is_morning_rush"] | out["is_evening_rush"]).astype(int)
    out["slot_of_day_ratio"] = out["slot_idx"] / 96.0

    # Distance to nearest rush boundary (vectorised)
    t = out["time_in_minutes"].values.astype(float)
    d1 = np.abs(t - 420)
    d2 = np.abs(t - 599)
    d3 = np.abs(t - 1020)
    d4 = np.abs(t - 1199)
    min_d = np.minimum(np.minimum(d1, d2), np.minimum(d3, d4))
    min_d[(t >= 420) & (t <= 599)] = 0
    min_d[(t >= 1020) & (t <= 1199)] = 0
    out["time_dist_to_rush"] = min_d

    # Geohash spatial
    coords = [_decode(g) for g in out["geohash"]]
    out["latitude"] = [c[0] for c in coords]
    out["longitude"] = [c[1] for c in coords]
    out["lat_lon_interaction"] = out["latitude"] * out["longitude"]

    mean_lat = out["latitude"].mean()
    mean_lon = out["longitude"].mean()
    out["dist_to_centroid"] = np.sqrt(
        (out["latitude"] - mean_lat) ** 2 + (out["longitude"] - mean_lon) ** 2
    )

    # Multi-granularity prefixes
    out["geohash_prefix2"] = out["geohash"].str[:2]
    out["geohash_prefix3"] = out["geohash"].str[:3]
    out["geohash_prefix4"] = out["geohash"].str[:4]
    out["geohash_prefix5"] = out["geohash"].str[:5]

    # Geohash character ordinals (each char = spatial band)
    for i in range(6):
        out[f"gh_char{i}"] = out["geohash"].apply(
            lambda x, idx=i: B32_MAP.get(x[idx], 0) if len(x) > idx else 0
        )

    # Weather / road interactions
    wmap = {"Sunny": 0, "Foggy": 1, "Rainy": 2, "Snowy": 3, "Unknown": 0}
    out["weather_encoded"] = out["Weather"].map(wmap).fillna(0).astype(int)
    out["weather_temp"] = out["weather_encoded"] * out["Temperature"]
    out["temp_lane_interaction"] = out["Temperature"] * out["NumberofLanes"]
    out["hour_lane_interaction"] = out["hour"] * out["NumberofLanes"]
    out["large_vehicles_allowed"] = (out["LargeVehicles"] == "Allowed").astype(int)
    out["has_landmarks"] = (out["Landmarks"] == "Yes").astype(int)
    out["Temperature_sq"] = out["Temperature"] ** 2
    out["temp_x_hour"] = out["Temperature"] * out["hour"]
    out["lanes_x_rush"] = out["NumberofLanes"] * out["is_rush"]
    return out


# ============================================================
# Day-48 lag features
# ============================================================
def compute_d48_features(df, history, global_mean):
    out = df.copy()
    d48 = history[history["day"] == 48].copy()

    if len(d48) == 0:
        for c in [
            "demand_d48_same_slot", "demand_d48_prev_slot", "demand_d48_next_slot",
            "d48_roll3_mean", "d48_roll3_std",
            "d48_hour_mean", "d48_hour_std", "d48_hour_min", "d48_hour_max",
            "d48_hour_range", "d48_delta_from_gh_mean", "d48_ratio_to_gh_mean",
            "gh_hour_d48",
        ]:
            out[c] = global_mean if "ratio" not in c else 1.0
        return out

    d48 = d48.sort_values(["geohash", "slot_idx"])

    # Same-slot
    d48_same = (
        d48[["geohash", "timestamp", TARGET]]
        .drop_duplicates(subset=["geohash", "timestamp"])
        .rename(columns={TARGET: "demand_d48_same_slot"})
    )

    # ±1 slot
    d48["_prev"] = d48.groupby("geohash")[TARGET].shift(1)
    d48["_next"] = d48.groupby("geohash")[TARGET].shift(-1)
    d48_prev = (
        d48[["geohash", "timestamp", "_prev"]]
        .drop_duplicates(subset=["geohash", "timestamp"])
        .rename(columns={"_prev": "demand_d48_prev_slot"})
    )
    d48_next = (
        d48[["geohash", "timestamp", "_next"]]
        .drop_duplicates(subset=["geohash", "timestamp"])
        .rename(columns={"_next": "demand_d48_next_slot"})
    )

    # Rolling 3-slot mean/std (centred)
    d48["_rm"] = d48.groupby("geohash")[TARGET].transform(
        lambda s: s.rolling(3, min_periods=1, center=True).mean()
    )
    d48["_rs"] = d48.groupby("geohash")[TARGET].transform(
        lambda s: s.rolling(3, min_periods=1, center=True).std().fillna(0)
    )
    d48_roll = (
        d48[["geohash", "timestamp", "_rm", "_rs"]]
        .drop_duplicates(subset=["geohash", "timestamp"])
        .rename(columns={"_rm": "d48_roll3_mean", "_rs": "d48_roll3_std"})
    )

    # Hour-level stats from d48
    d48h = d48.groupby(["geohash", "hour"])[TARGET].agg(["mean", "std", "min", "max"]).reset_index()
    d48h.columns = ["geohash", "hour", "d48_hour_mean", "d48_hour_std", "d48_hour_min", "d48_hour_max"]
    d48h["d48_hour_std"] = d48h["d48_hour_std"].fillna(0)
    d48h["d48_hour_range"] = d48h["d48_hour_max"] - d48h["d48_hour_min"]

    # Geohash mean & geohash×hour mean from d48
    gh_mean_d48 = d48.groupby("geohash")[TARGET].mean()
    gh_hour_d48 = d48.groupby(["geohash", "hour"])[TARGET].mean().rename("gh_hour_d48").reset_index()

    # Merge everything
    out = out.merge(d48_same, on=["geohash", "timestamp"], how="left")
    out = out.merge(d48_prev, on=["geohash", "timestamp"], how="left")
    out = out.merge(d48_next, on=["geohash", "timestamp"], how="left")
    out = out.merge(d48_roll, on=["geohash", "timestamp"], how="left")
    out = out.merge(d48h, on=["geohash", "hour"], how="left")
    out = out.merge(gh_hour_d48, on=["geohash", "hour"], how="left")

    # Delta / ratio
    out["_gh_mean"] = out["geohash"].map(gh_mean_d48)
    out["d48_delta_from_gh_mean"] = out["demand_d48_same_slot"] - out["_gh_mean"]
    out["d48_ratio_to_gh_mean"] = out["demand_d48_same_slot"] / (out["_gh_mean"] + 1e-9)

    # For d48 rows themselves, use geohash-hour proxy (avoid leaking own label)
    is_d48 = out["day"] == 48
    out.loc[is_d48, "demand_d48_same_slot"] = out.loc[is_d48, "gh_hour_d48"]

    # Fill NaN cascades
    out["demand_d48_same_slot"] = out["demand_d48_same_slot"].fillna(out["gh_hour_d48"]).fillna(global_mean)
    out["demand_d48_prev_slot"] = out["demand_d48_prev_slot"].fillna(out["demand_d48_same_slot"])
    out["demand_d48_next_slot"] = out["demand_d48_next_slot"].fillna(out["demand_d48_same_slot"])
    out["d48_roll3_mean"] = out["d48_roll3_mean"].fillna(out["demand_d48_same_slot"])
    out["d48_roll3_std"] = out["d48_roll3_std"].fillna(0)
    out["d48_hour_mean"] = out["d48_hour_mean"].fillna(out["demand_d48_same_slot"])
    out["d48_hour_std"] = out["d48_hour_std"].fillna(0)
    out["d48_hour_min"] = out["d48_hour_min"].fillna(out["demand_d48_same_slot"])
    out["d48_hour_max"] = out["d48_hour_max"].fillna(out["demand_d48_same_slot"])
    out["d48_hour_range"] = out["d48_hour_range"].fillna(0)
    out["gh_hour_d48"] = out["gh_hour_d48"].fillna(global_mean)
    out["d48_delta_from_gh_mean"] = out["d48_delta_from_gh_mean"].fillna(0)
    out["d48_ratio_to_gh_mean"] = out["d48_ratio_to_gh_mean"].fillna(1.0)
    return out.drop(columns=["_gh_mean"], errors="ignore")


# ============================================================
# Target encodings (Bayesian smoothed)
# ============================================================
def compute_target_encodings(df, history, global_mean, m=SMOOTHING_M):
    out = df.copy()
    h = history.reset_index(drop=True)
    y_vals = h[TARGET].values

    def _enc(hist_keys, out_keys, col_name):
        tmp = pd.DataFrame({"key": hist_keys, "target": y_vals})
        g = tmp.groupby("key")["target"].agg(["mean", "count"])
        smoothed = (g["count"] * g["mean"] + m * global_mean) / (g["count"] + m)
        out[col_name] = out_keys.map(smoothed.to_dict()).fillna(global_mean)

    # Geohash
    _enc(h["geohash"].values, out["geohash"], "enc_geohash_mean")
    out["enc_geohash_std"] = out["geohash"].map(
        h.groupby("geohash")[TARGET].std().to_dict()
    ).fillna(0)

    # Geohash × hour
    _enc(
        (h["geohash"] + "|" + h["hour"].astype(str)).values,
        out["geohash"] + "|" + out["hour"].astype(str),
        "enc_geohash_hour_mean",
    )

    # Geohash × slot
    _enc(
        (h["geohash"] + "|" + h["slot_idx"].astype(str)).values,
        out["geohash"] + "|" + out["slot_idx"].astype(str),
        "enc_geohash_slot_mean",
    )

    # Prefix-level
    for pl in [3, 4, 5]:
        _enc(
            h["geohash"].str[:pl].values,
            out["geohash"].str[:pl],
            f"enc_prefix{pl}_mean",
        )

    # RoadType, RoadType×hour
    _enc(h["RoadType"].values, out["RoadType"], "enc_roadtype_mean")
    _enc(
        (h["RoadType"].astype(str) + "|" + h["hour"].astype(str)).values,
        out["RoadType"].astype(str) + "|" + out["hour"].astype(str),
        "enc_roadtype_hour_mean",
    )

    # Weather, Weather×hour
    _enc(h["Weather"].values, out["Weather"], "enc_weather_mean")
    _enc(
        (h["Weather"].astype(str) + "|" + h["hour"].astype(str)).values,
        out["Weather"].astype(str) + "|" + out["hour"].astype(str),
        "enc_weather_hour_mean",
    )

    # Timestamp
    _enc(h["timestamp"].astype(str).values, out["timestamp"].astype(str), "enc_timestamp_mean")
    return out


# ============================================================
# Label encoding for LGB / XGB
# ============================================================
def label_encode(tr, va, te, cols):
    tr, va, te = tr.copy(), va.copy(), te.copy()
    for col in cols:
        if col not in tr.columns:
            continue
        combined = pd.concat([tr[col], va[col], te[col]]).astype(str)
        mapping = {k: v for v, k in enumerate(combined.unique())}
        tr[col] = tr[col].astype(str).map(mapping).fillna(-1).astype(int)
        va[col] = va[col].astype(str).map(mapping).fillna(-1).astype(int)
        te[col] = te[col].astype(str).map(mapping).fillna(-1).astype(int)
    return tr, va, te


# ============================================================
# Hour calibration from d49 train rows
# ============================================================
def hour_calibration(history, global_mean):
    d49 = history[history["day"] == 49].copy()
    d48 = history[history["day"] == 48].copy()
    if len(d49) == 0 or len(d48) == 0:
        return pd.Series(dtype=float), 1.0
    d48_lk = d48[["geohash", "timestamp", TARGET]].rename(
        columns={TARGET: "_d48d"}
    ).drop_duplicates(subset=["geohash", "timestamp"])
    d49 = d49.merge(d48_lk, on=["geohash", "timestamp"], how="left")
    d49["_d48d"] = d49["_d48d"].fillna(global_mean)
    valid = d49["_d48d"] > 1e-9
    d49 = d49[valid]
    d49["_ratio"] = d49[TARGET] / d49["_d48d"]
    hour_ratio = d49.groupby("hour")["_ratio"].median()
    return hour_ratio, float(d49["_ratio"].median())


# ============================================================
# Feature column list
# ============================================================
FEATURE_COLS = [
    # Spatial
    "latitude", "longitude", "lat_lon_interaction", "dist_to_centroid",
    "gh_char0", "gh_char1", "gh_char2", "gh_char3", "gh_char4", "gh_char5",
    # Temporal
    "day", "hour", "minute", "slot_idx", "time_in_minutes",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "slot_sin", "slot_cos", "day_sin", "day_cos",
    "is_morning_rush", "is_evening_rush", "is_night", "is_midday", "is_rush",
    "slot_of_day_ratio", "time_dist_to_rush",
    # Road / weather
    "NumberofLanes", "Temperature", "weather_encoded", "weather_temp",
    "temp_lane_interaction", "hour_lane_interaction",
    "large_vehicles_allowed", "has_landmarks",
    "Temperature_sq", "temp_x_hour", "lanes_x_rush",
    # Day-48 lags
    "demand_d48_same_slot", "demand_d48_prev_slot", "demand_d48_next_slot",
    "d48_roll3_mean", "d48_roll3_std",
    "d48_hour_mean", "d48_hour_std", "d48_hour_min", "d48_hour_max", "d48_hour_range",
    "d48_delta_from_gh_mean", "d48_ratio_to_gh_mean", "gh_hour_d48",
    # Target encodings
    "enc_geohash_mean", "enc_geohash_std",
    "enc_geohash_hour_mean", "enc_geohash_slot_mean",
    "enc_prefix3_mean", "enc_prefix4_mean", "enc_prefix5_mean",
    "enc_roadtype_mean", "enc_roadtype_hour_mean",
    "enc_weather_mean", "enc_weather_hour_mean",
    "enc_timestamp_mean",
    # Categoricals
    "geohash", "RoadType", "LargeVehicles", "Landmarks",
    "Weather", "timestamp", "geohash_prefix4",
]


# ============================================================
# Main pipeline
# ============================================================
def run(n_folds: int = N_FOLDS) -> dict:
    print("=" * 60)
    print("TRAIN V4 — 3-Model Stacking + Advanced Features (KFold)")
    print("=" * 60)

    train, test = load_data()
    impute(train, test)
    train = create_base_features(train)
    test = create_base_features(test)

    y = train[TARGET].values
    global_mean = float(y.mean())
    print(f"Train {len(train)} rows, Test {len(test)} rows, global mean={global_mean:.6f}")

    # Full-data d48 + target enc for test
    test_full = compute_d48_features(test, train, global_mean)
    test_full = compute_target_encodings(test_full, train, global_mean)

    # Hour calibration
    hour_ratio, global_ratio = hour_calibration(train, global_mean)
    test_full["calibrated_d48"] = (
        test_full["demand_d48_same_slot"]
        * test_full["hour"].map(hour_ratio).fillna(global_ratio)
    )

    # Ensure all feature cols present in test
    for c in FEATURE_COLS:
        if c not in test_full.columns:
            test_full[c] = 0

    X_test_base = test_full[FEATURE_COLS].copy()
    X_test_cb = X_test_base.copy()
    for c in CAT_FEATURES:
        if c in X_test_cb.columns:
            X_test_cb[c] = X_test_cb[c].astype(str)

    # ── KFold ──────────────────────────────────────────────────
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof_cat = np.zeros(len(train))
    oof_lgb = np.zeros(len(train))
    oof_xgb = np.zeros(len(train))
    oof_d48 = np.zeros(len(train))
    test_cat = np.zeros(len(test))
    test_lgb = np.zeros(len(test))
    test_xgb = np.zeros(len(test))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train)):
        print(f"\n{'=' * 55}\n  FOLD {fold + 1}/{n_folds}\n{'=' * 55}")

        tr_df = train.iloc[tr_idx].copy()
        va_df = train.iloc[va_idx].copy()

        # D48 + target encodings (from train fold only → no leakage)
        tr_feat = compute_d48_features(tr_df, tr_df, global_mean)
        va_feat = compute_d48_features(va_df, tr_df, global_mean)
        tr_feat = compute_target_encodings(tr_feat, tr_df, global_mean)
        va_feat = compute_target_encodings(va_feat, tr_df, global_mean)

        for c in FEATURE_COLS:
            if c not in tr_feat.columns:
                tr_feat[c] = 0
            if c not in va_feat.columns:
                va_feat[c] = 0

        X_tr = tr_feat[FEATURE_COLS].copy()
        X_va = va_feat[FEATURE_COLS].copy()
        y_tr, y_va = y[tr_idx], y[va_idx]

        oof_d48[va_idx] = va_feat["demand_d48_same_slot"].values

        # ── CatBoost ───────────────────────────────────────────
        for c in CAT_FEATURES:
            X_tr[c] = X_tr[c].astype(str)
            X_va[c] = X_va[c].astype(str)

        cb = CatBoostRegressor(
            iterations=5000, learning_rate=0.03, depth=8,
            l2_leaf_reg=7, random_strength=2, bagging_temperature=0.8,
            loss_function="RMSE", eval_metric="R2",
            random_seed=RANDOM_SEED + fold, verbose=500,
        )
        cb.fit(
            X_tr, y_tr,
            cat_features=[c for c in CAT_FEATURES if c in X_tr.columns],
            eval_set=(X_va, y_va),
            early_stopping_rounds=300, use_best_model=True,
        )
        oof_cat[va_idx] = cb.predict(X_va)
        test_cat += cb.predict(X_test_cb) / n_folds

        # ── LightGBM (label-encoded) ──────────────────────────
        X_tr_le, X_va_le, X_te_le = label_encode(
            tr_feat[FEATURE_COLS], va_feat[FEATURE_COLS],
            test_full[FEATURE_COLS], CAT_FEATURES,
        )
        lgb_model = LGBMRegressor(
            n_estimators=5000, learning_rate=0.02,
            num_leaves=127, max_depth=-1, min_child_samples=50,
            subsample=0.85, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=2.0,
            random_state=RANDOM_SEED + fold, n_jobs=-1, verbose=-1,
        )
        lgb_model.fit(
            X_tr_le, y_tr,
            eval_set=[(X_va_le, y_va)],
            callbacks=[early_stopping(300, verbose=False)],
        )
        oof_lgb[va_idx] = lgb_model.predict(X_va_le)
        test_lgb += lgb_model.predict(X_te_le) / n_folds

        # ── XGBoost (same label-encoded data) ─────────────────
        xgb_m = XGBRegressor(
            n_estimators=5000, learning_rate=0.03, max_depth=8,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=2.0,
            tree_method="hist", early_stopping_rounds=300,
            random_state=RANDOM_SEED + fold, verbosity=0,
        )
        xgb_m.fit(X_tr_le, y_tr, eval_set=[(X_va_le, y_va)], verbose=False)
        oof_xgb[va_idx] = xgb_m.predict(X_va_le)
        test_xgb += xgb_m.predict(X_te_le) / n_folds

        print(f"  CatBoost R²: {r2_score(y_va, oof_cat[va_idx]):.6f}")
        print(f"  LightGBM R²: {r2_score(y_va, oof_lgb[va_idx]):.6f}")
        print(f"  XGBoost  R²: {r2_score(y_va, oof_xgb[va_idx]):.6f}")

    # ── Meta-learner ───────────────────────────────────────────
    print(f"\n{'=' * 60}\n  META-LEARNER (Ridge Stacking)\n{'=' * 60}")
    print(f"CatBoost  OOF R²: {r2_score(y, oof_cat):.6f}")
    print(f"LightGBM  OOF R²: {r2_score(y, oof_lgb):.6f}")
    print(f"XGBoost   OOF R²: {r2_score(y, oof_xgb):.6f}")
    print(f"V3-style  OOF R²: {r2_score(y, 0.7 * oof_cat + 0.3 * oof_lgb):.6f}")
    print(f"Avg 3     OOF R²: {r2_score(y, (oof_cat + oof_lgb + oof_xgb) / 3):.6f}")

    test_d48_vals = test_full["demand_d48_same_slot"].values
    meta_tr = np.column_stack([oof_cat, oof_lgb, oof_xgb, oof_d48])
    meta_te = np.column_stack([test_cat, test_lgb, test_xgb, test_d48_vals])

    ridge = Ridge(alpha=1.0)
    ridge.fit(meta_tr, y)
    oof_stacked = ridge.predict(meta_tr)
    test_stacked = ridge.predict(meta_te)
    stacked_r2 = r2_score(y, oof_stacked)

    print(f"Stacked   OOF R²: {stacked_r2:.6f}")
    w = ridge.coef_
    print(f"  weights → CB={w[0]:.4f}  LGB={w[1]:.4f}  XGB={w[2]:.4f}  D48={w[3]:.4f}")

    # ── Hour-calibration blend ─────────────────────────────────
    cal_d48_test = test_full["calibrated_d48"].values
    best_wm, best_wc, best_blend_r2 = 1.0, 0.0, stacked_r2

    d49_mask = train["day"].values == 49
    if d49_mask.sum() > 0:
        d49_stk = oof_stacked[d49_mask]
        d49_y = y[d49_mask]
        d49_d48 = oof_d48[d49_mask]
        d49_cal = d49_d48 * train.loc[d49_mask, "hour"].map(hour_ratio).fillna(global_ratio).values
        for wm in np.arange(0.50, 1.01, 0.05):
            wc = round(1.0 - wm, 2)
            r2 = r2_score(d49_y, wm * d49_stk + wc * d49_cal)
            if r2 > best_blend_r2:
                best_blend_r2, best_wm, best_wc = r2, wm, wc
        print(f"Blend optimised: model={best_wm:.2f}  cal_d48={best_wc:.2f}  R²={best_blend_r2:.6f}")

    final = np.clip(best_wm * test_stacked + best_wc * cal_d48_test, 0, None)

    # ── Geohash calibration ────────────────────────────────────
    oof_df = pd.DataFrame({"gh": train["geohash"].values, "p": oof_stacked, "t": y})
    oof_df = oof_df[oof_df["p"] > 1e-9].copy()
    oof_df["ratio"] = oof_df["t"] / oof_df["p"]
    gh_stats = oof_df.groupby("gh")["ratio"].agg(["mean", "count"])
    gh_global = float(oof_df["ratio"].mean())
    gh_cal = (gh_stats["count"] * gh_stats["mean"] + 20 * gh_global) / (gh_stats["count"] + 20)

    cal_factors = test["geohash"].map(gh_cal.to_dict()).fillna(gh_global).values
    final_cal = np.clip(final * cal_factors, 0, None)

    if d49_mask.sum() > 0:
        cf_tr = pd.Series(train["geohash"].values).map(gh_cal.to_dict()).fillna(gh_global).values
        r2_uc = r2_score(y[d49_mask], oof_stacked[d49_mask])
        r2_c = r2_score(y[d49_mask], (oof_stacked * cf_tr)[d49_mask])
        print(f"Geohash calibration: uncal={r2_uc:.6f}  cal={r2_c:.6f}")
        use_final = final_cal if r2_c > r2_uc else final
    else:
        use_final = final

    # ── Save ───────────────────────────────────────────────────
    submission = pd.DataFrame({"Index": test["Index"], "demand": use_final})
    out_path = OUTPUT_DIR / "submission_v4.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)

    meta = {
        "oof_r2_catboost": float(r2_score(y, oof_cat)),
        "oof_r2_lightgbm": float(r2_score(y, oof_lgb)),
        "oof_r2_xgboost": float(r2_score(y, oof_xgb)),
        "oof_r2_stacked": float(stacked_r2),
        "blend_model": float(best_wm),
        "blend_cal_d48": float(best_wc),
        "ridge_weights": dict(zip(["catboost", "lightgbm", "xgboost", "d48"], w.tolist())),
        "n_folds": n_folds,
        "n_features": len(FEATURE_COLS),
    }
    (OUTPUT_DIR / "submission_v4_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nSaved {out_path} ({len(submission)} rows)")
    print(submission["demand"].describe())
    return meta


def main():
    parser = argparse.ArgumentParser(description="Train v4 — 3-model stacking pipeline")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    args = parser.parse_args()
    run(n_folds=args.folds)


if __name__ == "__main__":
    main()
