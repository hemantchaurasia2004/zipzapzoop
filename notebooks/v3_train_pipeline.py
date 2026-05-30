"""
Traffic demand pipeline v3 — self-contained script for Google Colab/local run.
"""
import os
import sys
import json
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

# Setup paths (dynamic lookup for Local vs Google Colab)
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    print("Running in Google Colab environment.")
    # Auto-install python-geohash and catboost if missing in Colab
    os.system("pip install -q python-geohash catboost")
    
    if Path("Grid-Flipkart/dataset").exists():
        ROOT = Path("Grid-Flipkart")
    elif Path("../dataset").exists():
        ROOT = Path("..")
    else:
        ROOT = Path(".")
else:
    ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

DATA_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "outputs"
RANDOM_SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "demand"
N_FOLDS = 5
CAT_FEATURES = [
    "geohash",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather",
    "timestamp",
]


def load_data():
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Dataset files not found at {DATA_DIR}. "
            "Please upload train.csv and test.csv."
        )
        
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train.columns = train.columns.str.strip()
    test.columns = test.columns.str.strip()
    return train, test


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = df["timestamp"].astype(str)
    df["hour"] = df["timestamp"].apply(lambda x: int(x.split(":")[0]))
    df["minute"] = df["timestamp"].apply(lambda x: int(x.split(":")[1]))
    df["time_in_minutes"] = df["hour"] * 60 + df["minute"]
    df["slot_idx"] = df["hour"] * 4 + df["minute"] // 15

    df["is_morning_rush"] = df["hour"].isin([7, 8, 9]).astype(int)
    df["is_evening_rush"] = df["hour"].isin([17, 18, 19]).astype(int)
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 7)

    lats, lons = [], []
    for gh in df["geohash"]:
        try:
            lat, lon = geohash.decode(gh)
            lats.append(float(lat))
            lons.append(float(lon))
        except Exception:
            lats.append(0.0)
            lons.append(0.0)
    df["latitude"] = lats
    df["longitude"] = lons

    df["lat_lon_interaction"] = df["latitude"] * df["longitude"]
    df["temp_lane_interaction"] = df["Temperature"] * df["NumberofLanes"]
    df["hour_lane_interaction"] = df["hour"] * df["NumberofLanes"]

    weather_map = {"Sunny": 0, "Foggy": 1, "Rainy": 2, "Snowy": 3}
    df["weather_encoded"] = df["Weather"].map(weather_map).fillna(0)
    df["weather_temp"] = df["weather_encoded"] * df["Temperature"]
    df["large_vehicles_allowed"] = (df["LargeVehicles"] == "Allowed").astype(int)
    df["has_landmarks"] = (df["Landmarks"] == "Yes").astype(int)
    return df


def impute(train: pd.DataFrame, test: pd.DataFrame):
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


def attach_d48_features(df: pd.DataFrame, history: pd.DataFrame, global_mean: float) -> pd.DataFrame:
    """Cross-day d48 lag for day 49; geohash-hour proxy for day 48 (no leakage)."""
    out = df.copy()
    d48 = history[history["day"] == 48]
    d48_map = d48.groupby(["geohash", "timestamp"])[TARGET].first()
    gh_hour = d48.groupby(["geohash", "hour"])[TARGET].mean()

    out = out.merge(
        d48_map.rename("demand_d48").reset_index(),
        on=["geohash", "timestamp"],
        how="left",
    )
    gh_hour_df = gh_hour.rename("gh_hour_d48").reset_index()
    out = out.merge(gh_hour_df, on=["geohash", "hour"], how="left")

    is_d48 = out["day"] == 48
    out.loc[is_d48, "demand_d48"] = out.loc[is_d48, "gh_hour_d48"]
    out["demand_d48"] = out["demand_d48"].fillna(out["gh_hour_d48"]).fillna(global_mean)
    out["log_d48"] = np.log1p(out["demand_d48"].clip(lower=0))
    return out.drop(columns=["gh_hour_d48"])


def _hour_calibration(history: pd.DataFrame) -> tuple[pd.Series, float]:
    d49 = attach_d48_features(history[history["day"] == 49], history, float(history[TARGET].mean()))
    valid = d49["demand_d48"] > 1e-9
    d49 = d49[valid].copy()
    d49["ratio"] = d49[TARGET] / d49["demand_d48"]
    hour_ratio = d49.groupby("hour")["ratio"].median()
    global_ratio = float(d49["ratio"].median())
    return hour_ratio, global_ratio


def label_encode(train_part, valid_part, test_part, cols):
    tr, va, te = train_part.copy(), valid_part.copy(), test_part.copy()
    for col in cols:
        combined = pd.concat([tr[col], va[col], te[col]]).astype(str)
        mapping = {k: v for v, k in enumerate(combined.unique())}
        tr[col] = tr[col].astype(str).map(mapping)
        va[col] = va[col].astype(str).map(mapping).fillna(-1).astype(int)
        te[col] = te[col].astype(str).map(mapping).fillna(-1).astype(int)
    return tr, va, te


def main():
    print(f"Project ROOT: {ROOT.resolve()}")
    print(f"DATA_DIR: {DATA_DIR.resolve()}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR.resolve()}")

    train, test = load_data()
    impute(train, test)
    train = create_features(train)
    test = create_features(test)

    y = train[TARGET].values
    global_mean = float(y.mean())

    # Full-train d48 for test + hour calibration
    hour_ratio, global_ratio = _hour_calibration(train)
    test = attach_d48_features(test, train, global_mean)
    test["hour_ratio"] = test["hour"].map(hour_ratio).fillna(global_ratio)
    test["calibrated_d48"] = test["demand_d48"] * test["hour_ratio"]

    feature_cols = [c for c in train.columns if c not in (TARGET, "Index")]
    for extra in ("demand_d48", "log_d48"):
        if extra not in feature_cols:
            feature_cols.append(extra)

    X_test = test[feature_cols].copy()
    for col in CAT_FEATURES:
        X_test[col] = X_test[col].astype(str)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(train))
    test_cat = np.zeros(len(test))
    test_lgb = np.zeros(len(test))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train)):
        print(f"\n{'=' * 50}\nFold {fold + 1}/{N_FOLDS}\n{'=' * 50}")

        tr = attach_d48_features(train.iloc[tr_idx], train.iloc[tr_idx], global_mean)
        va = attach_d48_features(train.iloc[va_idx], train.iloc[tr_idx], global_mean)

        X_tr = tr[feature_cols].copy()
        X_va = va[feature_cols].copy()
        for col in CAT_FEATURES:
            X_tr[col] = X_tr[col].astype(str)
            X_va[col] = X_va[col].astype(str)

        cat_model = CatBoostRegressor(
            iterations=3000,
            learning_rate=0.03,
            depth=8,
            loss_function="RMSE",
            eval_metric="R2",
            random_seed=RANDOM_SEED + fold,
            verbose=300,
        )
        cat_model.fit(
            X_tr,
            y[tr_idx],
            cat_features=CAT_FEATURES,
            eval_set=(X_va, y[va_idx]),
            early_stopping_rounds=200,
            use_best_model=True,
        )
        va_cat = cat_model.predict(X_va)
        test_cat += cat_model.predict(X_test) / N_FOLDS

        X_tr_l, X_va_l, X_te_l = label_encode(
            tr[feature_cols], va[feature_cols], test[feature_cols], CAT_FEATURES
        )
        lgb_model = LGBMRegressor(
            n_estimators=3000,
            learning_rate=0.02,
            num_leaves=128,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED + fold,
            verbose=-1,
        )
        lgb_model.fit(
            X_tr_l,
            y[tr_idx],
            eval_set=[(X_va_l, y[va_idx])],
            eval_metric="l2",
            callbacks=[early_stopping(200, verbose=False)],
        )
        va_lgb = lgb_model.predict(X_va_l)
        test_lgb += lgb_model.predict(X_te_l) / N_FOLDS

        va_ens = 0.7 * va_cat + 0.3 * va_lgb
        oof[va_idx] = va_ens
        print(f"Fold R2: {r2_score(y[va_idx], va_ens):.6f}")

    oof_r2 = r2_score(y, oof)
    print(f"\nOOF R2 (full model): {oof_r2:.6f}")

    test_model = 0.7 * test_cat + 0.3 * test_lgb

    # Optimize blend
    d49_mask = train["day"].values == 49
    if d49_mask.sum() > 0:
        d49_oof = oof[d49_mask]
        d49_y = y[d49_mask]
        d49_cal = attach_d48_features(train.loc[d49_mask], train, global_mean)
        d49_cal_pred = d49_cal["demand_d48"].values * d49_cal["hour"].map(hour_ratio).fillna(global_ratio).values

        best_r2, best_wm, best_wc = -1, 0.7, 0.3
        for wm in np.arange(0.5, 1.0, 0.05):
            for wc in np.arange(0.0, 0.51, 0.05):
                if wm + wc > 1.01:
                    continue
                pred = wm * d49_oof + wc * d49_cal_pred
                r2 = r2_score(d49_y, pred)
                if r2 > best_r2:
                    best_r2, best_wm, best_wc = r2, wm, wc
        print(f"Day-49 proxy blend: model={best_wm:.2f}, calibrated_d48={best_wc:.2f}, R2={best_r2:.6f}")
    else:
        best_wm, best_wc, best_r2 = 0.85, 0.15, oof_r2

    final = np.clip(best_wm * test_model + best_wc * test["calibrated_d48"].values, 0, 1.0)

    submission = pd.DataFrame({"Index": test["Index"], "demand": final})
    submission.to_csv(OUTPUT_DIR / "submission_v3.csv", index=False)
    print(f"Saved submission ({len(submission)} rows)")


if __name__ == "__main__":
    main()
