# ============================================================
# TRAFFIC DEMAND PREDICTION V5 — TEMPORAL SPLIT + 3-MODEL STACKING
# Colab-ready standalone script.  Run in a single notebook cell.
#
# Strategy:
#   1. Train on day 48, validate on day 49 early slots (0:00-2:00)
#   2. 3-model stacking: CatBoost + LightGBM + XGBoost
#   3. Grid-search optimal blend weights on validation
#   4. Retrain on ALL training data, predict test
#   5. Advanced d48 lag features + multi-granularity target encodings
# ============================================================

# INSTALL DEPENDENCIES (uncomment if running on Colab)
# !pip install -q catboost lightgbm xgboost pygeohash

# ============================================================
# IMPORTS
# ============================================================
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
import pygeohash as pgh

# ============================================================
# CONFIG
# ============================================================
RANDOM_SEED = 42
TARGET = "demand"
SMOOTHING_M = 20
CAT_FEATURES = [
    "geohash", "RoadType", "LargeVehicles", "Landmarks",
    "Weather", "timestamp", "geohash_prefix4",
]

BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
B32_MAP = {c: i for i, c in enumerate(BASE32)}

# ============================================================
# LOAD DATA
# ============================================================
train = pd.read_csv("/content/train.csv")
test  = pd.read_csv("/content/test.csv")
train.columns = train.columns.str.strip()
test.columns  = test.columns.str.strip()
print(f"Train: {train.shape}  |  Test: {test.shape}")

# ============================================================
# MISSING VALUES
# ============================================================
for col in train.columns:
    if col == TARGET:
        continue
    if train[col].dtype == "object":
        train[col] = train[col].replace("", np.nan).fillna("Unknown")
        test[col]  = test[col].replace("", np.nan).fillna("Unknown")
    else:
        med = train[col].median()
        train[col] = train[col].fillna(med)
        test[col]  = test[col].fillna(med)

# ============================================================
# GEOHASH DECODE
# ============================================================
def decode_geohash(g):
    try:
        lat, lon = pgh.decode(g)
        return float(lat), float(lon)
    except Exception:
        return 0.0, 0.0

# ============================================================
# BASE FEATURE ENGINEERING
# ============================================================
def create_base_features(df):
    out = df.copy()
    out["timestamp"] = out["timestamp"].astype(str)
    out["hour"]   = out["timestamp"].apply(lambda x: int(x.split(":")[0]))
    out["minute"] = out["timestamp"].apply(lambda x: int(x.split(":")[1]))
    out["time_in_minutes"] = out["hour"] * 60 + out["minute"]
    out["slot_idx"] = out["hour"] * 4 + out["minute"] // 15

    # Cyclical
    out["hour_sin"]   = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"]   = np.cos(2 * np.pi * out["hour"] / 24)
    out["minute_sin"] = np.sin(2 * np.pi * out["minute"] / 60)
    out["minute_cos"] = np.cos(2 * np.pi * out["minute"] / 60)
    out["slot_sin"]   = np.sin(2 * np.pi * out["slot_idx"] / 96)
    out["slot_cos"]   = np.cos(2 * np.pi * out["slot_idx"] / 96)
    out["day_sin"]    = np.sin(2 * np.pi * out["day"] / 7)
    out["day_cos"]    = np.cos(2 * np.pi * out["day"] / 7)

    # Time-of-day flags
    out["is_morning_rush"] = out["hour"].isin([7, 8, 9]).astype(int)
    out["is_evening_rush"] = out["hour"].isin([17, 18, 19]).astype(int)
    out["is_night"]        = out["hour"].isin([0, 1, 2, 3, 4]).astype(int)
    out["is_midday"]       = out["hour"].between(10, 14).astype(int)
    out["is_rush"]         = (out["is_morning_rush"] | out["is_evening_rush"]).astype(int)
    out["slot_of_day_ratio"] = out["slot_idx"] / 96.0

    # Distance to nearest rush boundary (vectorised)
    t = out["time_in_minutes"].values.astype(float)
    d1 = np.abs(t - 420); d2 = np.abs(t - 599)
    d3 = np.abs(t - 1020); d4 = np.abs(t - 1199)
    md = np.minimum(np.minimum(d1, d2), np.minimum(d3, d4))
    md[(t >= 420) & (t <= 599)] = 0
    md[(t >= 1020) & (t <= 1199)] = 0
    out["time_dist_to_rush"] = md

    # Geohash spatial
    coords = [decode_geohash(g) for g in out["geohash"]]
    out["latitude"]  = [c[0] for c in coords]
    out["longitude"] = [c[1] for c in coords]
    out["lat_lon_interaction"] = out["latitude"] * out["longitude"]

    mean_lat = out["latitude"].mean()
    mean_lon = out["longitude"].mean()
    out["dist_to_centroid"] = np.sqrt(
        (out["latitude"] - mean_lat)**2 + (out["longitude"] - mean_lon)**2
    )

    # Multi-granularity prefixes
    out["geohash_prefix2"] = out["geohash"].str[:2]
    out["geohash_prefix3"] = out["geohash"].str[:3]
    out["geohash_prefix4"] = out["geohash"].str[:4]
    out["geohash_prefix5"] = out["geohash"].str[:5]

    # Geohash character ordinals
    for i in range(6):
        out[f"gh_char{i}"] = out["geohash"].apply(
            lambda x, idx=i: B32_MAP.get(x[idx], 0) if len(x) > idx else 0
        )

    # Weather / road
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

print("Creating base features …")
train = create_base_features(train)
test  = create_base_features(test)
print("Done.")

# ============================================================
# DAY-48 LAG FEATURES
# ============================================================
def compute_d48_features(df, history, global_mean):
    out = df.copy()
    d48 = history[history["day"] == 48].copy()

    if len(d48) == 0:
        for c in ["demand_d48_same_slot","demand_d48_prev_slot","demand_d48_next_slot",
                   "d48_roll3_mean","d48_roll3_std","d48_hour_mean","d48_hour_std",
                   "d48_hour_min","d48_hour_max","d48_hour_range",
                   "d48_delta_from_gh_mean","d48_ratio_to_gh_mean","gh_hour_d48"]:
            out[c] = global_mean if "ratio" not in c else 1.0
        return out

    d48 = d48.sort_values(["geohash", "slot_idx"])

    # Same-slot
    d48_same = (d48[["geohash","timestamp",TARGET]]
                .drop_duplicates(subset=["geohash","timestamp"])
                .rename(columns={TARGET:"demand_d48_same_slot"}))

    # ±1 slot
    d48["_prev"] = d48.groupby("geohash")[TARGET].shift(1)
    d48["_next"] = d48.groupby("geohash")[TARGET].shift(-1)
    d48_prev = (d48[["geohash","timestamp","_prev"]]
                .drop_duplicates(subset=["geohash","timestamp"])
                .rename(columns={"_prev":"demand_d48_prev_slot"}))
    d48_next = (d48[["geohash","timestamp","_next"]]
                .drop_duplicates(subset=["geohash","timestamp"])
                .rename(columns={"_next":"demand_d48_next_slot"}))

    # Rolling 3-slot mean/std (centred)
    d48["_rm"] = d48.groupby("geohash")[TARGET].transform(
        lambda s: s.rolling(3, min_periods=1, center=True).mean())
    d48["_rs"] = d48.groupby("geohash")[TARGET].transform(
        lambda s: s.rolling(3, min_periods=1, center=True).std().fillna(0))
    d48_roll = (d48[["geohash","timestamp","_rm","_rs"]]
                .drop_duplicates(subset=["geohash","timestamp"])
                .rename(columns={"_rm":"d48_roll3_mean","_rs":"d48_roll3_std"}))

    # Hour-level stats
    d48h = d48.groupby(["geohash","hour"])[TARGET].agg(["mean","std","min","max"]).reset_index()
    d48h.columns = ["geohash","hour","d48_hour_mean","d48_hour_std","d48_hour_min","d48_hour_max"]
    d48h["d48_hour_std"]   = d48h["d48_hour_std"].fillna(0)
    d48h["d48_hour_range"] = d48h["d48_hour_max"] - d48h["d48_hour_min"]

    gh_mean_d48 = d48.groupby("geohash")[TARGET].mean()
    gh_hour_d48 = d48.groupby(["geohash","hour"])[TARGET].mean().rename("gh_hour_d48").reset_index()

    # Merge
    out = out.merge(d48_same, on=["geohash","timestamp"], how="left")
    out = out.merge(d48_prev, on=["geohash","timestamp"], how="left")
    out = out.merge(d48_next, on=["geohash","timestamp"], how="left")
    out = out.merge(d48_roll, on=["geohash","timestamp"], how="left")
    out = out.merge(d48h,     on=["geohash","hour"],      how="left")
    out = out.merge(gh_hour_d48, on=["geohash","hour"],   how="left")

    out["_gh_mean"] = out["geohash"].map(gh_mean_d48)
    out["d48_delta_from_gh_mean"] = out["demand_d48_same_slot"] - out["_gh_mean"]
    out["d48_ratio_to_gh_mean"]   = out["demand_d48_same_slot"] / (out["_gh_mean"] + 1e-9)

    is_d48 = out["day"] == 48
    out.loc[is_d48, "demand_d48_same_slot"] = out.loc[is_d48, "gh_hour_d48"]

    out["demand_d48_same_slot"] = out["demand_d48_same_slot"].fillna(out["gh_hour_d48"]).fillna(global_mean)
    out["demand_d48_prev_slot"] = out["demand_d48_prev_slot"].fillna(out["demand_d48_same_slot"])
    out["demand_d48_next_slot"] = out["demand_d48_next_slot"].fillna(out["demand_d48_same_slot"])
    out["d48_roll3_mean"]  = out["d48_roll3_mean"].fillna(out["demand_d48_same_slot"])
    out["d48_roll3_std"]   = out["d48_roll3_std"].fillna(0)
    out["d48_hour_mean"]   = out["d48_hour_mean"].fillna(out["demand_d48_same_slot"])
    out["d48_hour_std"]    = out["d48_hour_std"].fillna(0)
    out["d48_hour_min"]    = out["d48_hour_min"].fillna(out["demand_d48_same_slot"])
    out["d48_hour_max"]    = out["d48_hour_max"].fillna(out["demand_d48_same_slot"])
    out["d48_hour_range"]  = out["d48_hour_range"].fillna(0)
    out["gh_hour_d48"]     = out["gh_hour_d48"].fillna(global_mean)
    out["d48_delta_from_gh_mean"] = out["d48_delta_from_gh_mean"].fillna(0)
    out["d48_ratio_to_gh_mean"]   = out["d48_ratio_to_gh_mean"].fillna(1.0)
    return out.drop(columns=["_gh_mean"], errors="ignore")

# ============================================================
# TARGET ENCODINGS (Bayesian smoothed)
# ============================================================
def compute_target_encodings(df, history, global_mean, m=SMOOTHING_M):
    out = df.copy()
    h = history.reset_index(drop=True)
    y_vals = h[TARGET].values

    def _enc(hist_keys, out_keys, col_name):
        tmp = pd.DataFrame({"key": hist_keys, "target": y_vals})
        g = tmp.groupby("key")["target"].agg(["mean","count"])
        smoothed = (g["count"] * g["mean"] + m * global_mean) / (g["count"] + m)
        out[col_name] = out_keys.map(smoothed.to_dict()).fillna(global_mean)

    _enc(h["geohash"].values, out["geohash"], "enc_geohash_mean")
    out["enc_geohash_std"] = out["geohash"].map(
        h.groupby("geohash")[TARGET].std().to_dict()
    ).fillna(0)

    _enc((h["geohash"]+"|"+h["hour"].astype(str)).values,
         out["geohash"]+"|"+out["hour"].astype(str), "enc_geohash_hour_mean")

    _enc((h["geohash"]+"|"+h["slot_idx"].astype(str)).values,
         out["geohash"]+"|"+out["slot_idx"].astype(str), "enc_geohash_slot_mean")

    for pl in [3, 4, 5]:
        _enc(h["geohash"].str[:pl].values,
             out["geohash"].str[:pl], f"enc_prefix{pl}_mean")

    _enc(h["RoadType"].values, out["RoadType"], "enc_roadtype_mean")
    _enc((h["RoadType"].astype(str)+"|"+h["hour"].astype(str)).values,
         out["RoadType"].astype(str)+"|"+out["hour"].astype(str), "enc_roadtype_hour_mean")

    _enc(h["Weather"].values, out["Weather"], "enc_weather_mean")
    _enc((h["Weather"].astype(str)+"|"+h["hour"].astype(str)).values,
         out["Weather"].astype(str)+"|"+out["hour"].astype(str), "enc_weather_hour_mean")

    _enc(h["timestamp"].astype(str).values, out["timestamp"].astype(str), "enc_timestamp_mean")
    return out

# ============================================================
# LABEL ENCODING FOR LGB / XGB
# ============================================================
def label_encode_pair(tr, te, cols):
    """Label-encode two DataFrames (train, test) consistently."""
    tr, te = tr.copy(), te.copy()
    for col in cols:
        if col not in tr.columns:
            continue
        combined = pd.concat([tr[col], te[col]]).astype(str)
        mapping = {k: v for v, k in enumerate(combined.unique())}
        tr[col] = tr[col].astype(str).map(mapping).fillna(-1).astype(int)
        te[col] = te[col].astype(str).map(mapping).fillna(-1).astype(int)
    return tr, te


def label_encode_triple(tr, va, te, cols):
    """Label-encode three DataFrames consistently."""
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
# FEATURE COLUMNS
# ============================================================
FEATURE_COLS = [
    # Spatial
    "latitude","longitude","lat_lon_interaction","dist_to_centroid",
    "gh_char0","gh_char1","gh_char2","gh_char3","gh_char4","gh_char5",
    # Temporal
    "day","hour","minute","slot_idx","time_in_minutes",
    "hour_sin","hour_cos","minute_sin","minute_cos",
    "slot_sin","slot_cos","day_sin","day_cos",
    "is_morning_rush","is_evening_rush","is_night","is_midday","is_rush",
    "slot_of_day_ratio","time_dist_to_rush",
    # Road / weather
    "NumberofLanes","Temperature","weather_encoded","weather_temp",
    "temp_lane_interaction","hour_lane_interaction",
    "large_vehicles_allowed","has_landmarks",
    "Temperature_sq","temp_x_hour","lanes_x_rush",
    # D48 lags
    "demand_d48_same_slot","demand_d48_prev_slot","demand_d48_next_slot",
    "d48_roll3_mean","d48_roll3_std",
    "d48_hour_mean","d48_hour_std","d48_hour_min","d48_hour_max","d48_hour_range",
    "d48_delta_from_gh_mean","d48_ratio_to_gh_mean","gh_hour_d48",
    # Target encodings
    "enc_geohash_mean","enc_geohash_std",
    "enc_geohash_hour_mean","enc_geohash_slot_mean",
    "enc_prefix3_mean","enc_prefix4_mean","enc_prefix5_mean",
    "enc_roadtype_mean","enc_roadtype_hour_mean",
    "enc_weather_mean","enc_weather_hour_mean",
    "enc_timestamp_mean",
    # Categoricals
    "geohash","RoadType","LargeVehicles","Landmarks",
    "Weather","timestamp","geohash_prefix4",
]

# ============================================================
# TEMPORAL SPLIT — VALIDATION PHASE
# ============================================================
print("\n" + "="*60)
print("PHASE 1: TEMPORAL VALIDATION  (train=d48, val=d49 early)")
print("="*60)

y_all = train[TARGET].values
global_mean = float(y_all.mean())

train_d48 = train[train["day"] == 48].copy()
val_d49   = train[train["day"] == 49].copy()
print(f"Training rows (d48): {len(train_d48)}  |  Validation rows (d49 early): {len(val_d49)}")

# D48 features + target encodings for validation split
tr_feat  = compute_d48_features(train_d48, train_d48, global_mean)
va_feat  = compute_d48_features(val_d49,   train_d48, global_mean)
tr_feat  = compute_target_encodings(tr_feat, train_d48, global_mean)
va_feat  = compute_target_encodings(va_feat, train_d48, global_mean)

for c in FEATURE_COLS:
    if c not in tr_feat.columns:
        tr_feat[c] = 0
    if c not in va_feat.columns:
        va_feat[c] = 0

X_tr = tr_feat[FEATURE_COLS].copy()
X_va = va_feat[FEATURE_COLS].copy()
y_tr = train_d48[TARGET].values
y_va = val_d49[TARGET].values

# ── CatBoost (validation phase) ─────────────────────────────
X_tr_cb = X_tr.copy()
X_va_cb = X_va.copy()
for c in CAT_FEATURES:
    X_tr_cb[c] = X_tr_cb[c].astype(str)
    X_va_cb[c] = X_va_cb[c].astype(str)

cb_val = CatBoostRegressor(
    iterations=5000, learning_rate=0.03, depth=8,
    l2_leaf_reg=7, random_strength=2, bagging_temperature=0.8,
    loss_function="RMSE", eval_metric="R2",
    random_seed=RANDOM_SEED, verbose=500,
)
cb_val.fit(
    X_tr_cb, y_tr,
    cat_features=[c for c in CAT_FEATURES if c in X_tr_cb.columns],
    eval_set=(X_va_cb, y_va),
    early_stopping_rounds=300, use_best_model=True,
)
va_cat = cb_val.predict(X_va_cb)

# ── LightGBM (validation phase) ─────────────────────────────
# Create a dummy test for label encoding (use test itself)
X_tr_le, X_va_le, _ = label_encode_triple(
    tr_feat[FEATURE_COLS], va_feat[FEATURE_COLS],
    va_feat[FEATURE_COLS],  # placeholder
    CAT_FEATURES,
)

lgb_val = lgb.LGBMRegressor(
    n_estimators=5000, learning_rate=0.02,
    num_leaves=127, max_depth=-1, min_child_samples=50,
    subsample=0.85, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=2.0,
    random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
)
lgb_val.fit(
    X_tr_le, y_tr,
    eval_set=[(X_va_le, y_va)],
    callbacks=[lgb.early_stopping(300, verbose=False)],
)
va_lgb = lgb_val.predict(X_va_le)

# ── XGBoost (validation phase) ──────────────────────────────
xgb_val = XGBRegressor(
    n_estimators=5000, learning_rate=0.03, max_depth=8,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=2.0,
    tree_method="hist", early_stopping_rounds=300,
    random_state=RANDOM_SEED, verbosity=0,
)
xgb_val.fit(X_tr_le, y_tr, eval_set=[(X_va_le, y_va)], verbose=False)
va_xgb = xgb_val.predict(X_va_le)

print(f"\nValidation CatBoost R²: {r2_score(y_va, va_cat):.6f}")
print(f"Validation LightGBM R²: {r2_score(y_va, va_lgb):.6f}")
print(f"Validation XGBoost  R²: {r2_score(y_va, va_xgb):.6f}")

# ── Grid-search optimal blend weights ────────────────────────
best_r2, best_w = -1.0, (1/3, 1/3, 1/3)
for w1 in np.arange(0, 1.05, 0.05):
    for w2 in np.arange(0, 1.05 - w1, 0.05):
        w3 = round(1.0 - w1 - w2, 2)
        if w3 < -0.01:
            continue
        pred = w1 * va_cat + w2 * va_lgb + w3 * va_xgb
        r2 = r2_score(y_va, pred)
        if r2 > best_r2:
            best_r2, best_w = r2, (round(w1, 2), round(w2, 2), round(w3, 2))

print(f"\nOptimal blend:  CB={best_w[0]}  LGB={best_w[1]}  XGB={best_w[2]}")
print(f"Blended validation R²: {best_r2:.6f}")

# Also try Ridge stacking on validation predictions
va_d48_vals = va_feat["demand_d48_same_slot"].values
meta_va = np.column_stack([va_cat, va_lgb, va_xgb, va_d48_vals])
ridge_val = Ridge(alpha=1.0)
ridge_val.fit(meta_va, y_va)
va_ridge = ridge_val.predict(meta_va)
print(f"Ridge validation R² (on-train, indicative): {r2_score(y_va, va_ridge):.6f}")

# ============================================================
# PHASE 2: RETRAIN ON ALL DATA + PREDICT TEST
# ============================================================
print("\n" + "="*60)
print("PHASE 2: FULL RETRAIN  (all train data → test prediction)")
print("="*60)

# Full-data d48 features + target encodings
train_full = compute_d48_features(train, train, global_mean)
train_full = compute_target_encodings(train_full, train, global_mean)
test_full  = compute_d48_features(test, train, global_mean)
test_full  = compute_target_encodings(test_full, train, global_mean)

for c in FEATURE_COLS:
    if c not in train_full.columns:
        train_full[c] = 0
    if c not in test_full.columns:
        test_full[c] = 0

X_full = train_full[FEATURE_COLS].copy()
X_test = test_full[FEATURE_COLS].copy()
y_full = y_all

# ── CatBoost (full retrain) ─────────────────────────────────
X_full_cb = X_full.copy()
X_test_cb = X_test.copy()
for c in CAT_FEATURES:
    X_full_cb[c] = X_full_cb[c].astype(str)
    X_test_cb[c] = X_test_cb[c].astype(str)

cb_full = CatBoostRegressor(
    iterations=5000, learning_rate=0.03, depth=8,
    l2_leaf_reg=7, random_strength=2, bagging_temperature=0.8,
    loss_function="RMSE", eval_metric="R2",
    random_seed=RANDOM_SEED, verbose=500,
)
cb_full.fit(X_full_cb, y_full, cat_features=[c for c in CAT_FEATURES if c in X_full_cb.columns])
test_cat_preds = cb_full.predict(X_test_cb)

# ── LightGBM (full retrain) ─────────────────────────────────
X_full_le, X_test_le = label_encode_pair(
    train_full[FEATURE_COLS], test_full[FEATURE_COLS], CAT_FEATURES,
)

lgb_full = lgb.LGBMRegressor(
    n_estimators=5000, learning_rate=0.02,
    num_leaves=127, max_depth=-1, min_child_samples=50,
    subsample=0.85, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=2.0,
    random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
)
lgb_full.fit(X_full_le, y_full)
test_lgb_preds = lgb_full.predict(X_test_le)

# ── XGBoost (full retrain) ──────────────────────────────────
xgb_full = XGBRegressor(
    n_estimators=5000, learning_rate=0.03, max_depth=8,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=2.0,
    tree_method="hist",
    random_state=RANDOM_SEED, verbosity=0,
)
xgb_full.fit(X_full_le, y_full)
test_xgb_preds = xgb_full.predict(X_test_le)

# ── Blend with optimised weights ─────────────────────────────
w1, w2, w3 = best_w
test_blended = w1 * test_cat_preds + w2 * test_lgb_preds + w3 * test_xgb_preds

# ── Hour calibration ─────────────────────────────────────────
print("\nApplying hour calibration …")
d49_train = train[train["day"] == 49].copy()
d48_lk = (train[train["day"] == 48][["geohash","timestamp",TARGET]]
           .drop_duplicates(subset=["geohash","timestamp"])
           .rename(columns={TARGET: "_d48d"}))
d49_cal = d49_train.merge(d48_lk, on=["geohash","timestamp"], how="left")
d49_cal["_d48d"] = d49_cal["_d48d"].fillna(global_mean)
valid_mask = d49_cal["_d48d"] > 1e-9
d49_cal = d49_cal[valid_mask]
d49_cal["_ratio"] = d49_cal[TARGET] / d49_cal["_d48d"]
hour_ratio = d49_cal.groupby("hour")["_ratio"].median()
global_ratio = float(d49_cal["_ratio"].median())

test_d48_vals = test_full["demand_d48_same_slot"].values
test_calibrated_d48 = test_d48_vals * test_full["hour"].map(hour_ratio).fillna(global_ratio).values

# Blend model vs calibrated d48
best_blend_r2_final = best_r2
best_wm_final = 1.0
# Use validation to optimise model vs d48 blend
va_d48_cal = va_feat["demand_d48_same_slot"].values * va_feat["hour"].map(hour_ratio).fillna(global_ratio).values
for wm in np.arange(0.50, 1.01, 0.05):
    wc = round(1.0 - wm, 2)
    va_blend = wm * (best_w[0] * va_cat + best_w[1] * va_lgb + best_w[2] * va_xgb) + wc * va_d48_cal
    r2 = r2_score(y_va, va_blend)
    if r2 > best_blend_r2_final:
        best_blend_r2_final = r2
        best_wm_final = wm

best_wc_final = round(1.0 - best_wm_final, 2)
print(f"Model vs D48 blend: model={best_wm_final:.2f}  cal_d48={best_wc_final:.2f}  R²={best_blend_r2_final:.6f}")

final_preds = best_wm_final * test_blended + best_wc_final * test_calibrated_d48
final_preds = np.clip(final_preds, 0, None)

# ============================================================
# SUBMISSION
# ============================================================
submission = pd.DataFrame({
    "Index": test["Index"],
    "demand": final_preds,
})

submission.to_csv("/content/submission_v5.csv", index=False)
print(f"\nsubmission_v5.csv created! ({len(submission)} rows)")
print("\nPREDICTION STATS:")
print(submission["demand"].describe())
print(f"\nValidation R² (temporal): {best_blend_r2_final:.6f}")
print(f"Leaderboard score estimate: {max(0, 100 * best_blend_r2_final):.2f}")
# ============================================================
# DONE
# ============================================================
