# ============================================================
# TRAFFIC DEMAND PREDICTION - CORRECTED HIGH R² PIPELINE
# ============================================================

# INSTALL REQUIRED LIBRARIES
!pip install -q catboost lightgbm pygeohash

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from catboost import CatBoostRegressor
import lightgbm as lgb
import pygeohash as pgh

# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv("/content/train.csv")
test = pd.read_csv("/content/test.csv")

# REMOVE HIDDEN SPACES FROM COLUMN NAMES
train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()

print("Train Shape:", train.shape)
print("Test Shape :", test.shape)

print("\nTRAIN COLUMNS:")
print(train.columns)

# ============================================================
# TARGET COLUMN
# ============================================================

TARGET = "demand"

# ============================================================
# HANDLE MISSING VALUES
# ============================================================

for col in train.columns:

    # Skip target
    if col == TARGET:
        continue

    if train[col].dtype == "object":

        train[col] = train[col].fillna("Unknown")
        test[col] = test[col].fillna("Unknown")

    else:

        median_value = train[col].median()

        train[col] = train[col].fillna(median_value)
        test[col] = test[col].fillna(median_value)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):

    # --------------------------------------------------------
    # TIME FEATURES
    # --------------------------------------------------------

    df["timestamp"] = df["timestamp"].astype(str)

    df["hour"] = df["timestamp"].apply(
        lambda x: int(x.split(":")[0])
    )

    df["minute"] = df["timestamp"].apply(
        lambda x: int(x.split(":")[1])
    )

    df["time_in_minutes"] = (
        df["hour"] * 60 + df["minute"]
    )

    # --------------------------------------------------------
    # RUSH HOUR FEATURES
    # --------------------------------------------------------

    df["is_morning_rush"] = (
        df["hour"].isin([7, 8, 9]).astype(int)
    )

    df["is_evening_rush"] = (
        df["hour"].isin([17, 18, 19]).astype(int)
    )

    df["is_night"] = (
        df["hour"].isin([0,1,2,3,4]).astype(int)
    )

    # --------------------------------------------------------
    # CYCLICAL ENCODING
    # --------------------------------------------------------

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["minute_sin"] = np.sin(
        2 * np.pi * df["minute"] / 60
    )

    df["minute_cos"] = np.cos(
        2 * np.pi * df["minute"] / 60
    )

    df["day_sin"] = np.sin(
        2 * np.pi * df["day"] / 7
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["day"] / 7
    )

    # --------------------------------------------------------
    # GEOHASH FEATURES
    # --------------------------------------------------------

    latitudes = []
    longitudes = []

    for gh in df["geohash"]:

        try:
            lat, lon = pgh.decode(gh)

            latitudes.append(float(lat))
            longitudes.append(float(lon))

        except:

            latitudes.append(0)
            longitudes.append(0)

    df["latitude"] = latitudes
    df["longitude"] = longitudes

    # --------------------------------------------------------
    # INTERACTION FEATURES
    # --------------------------------------------------------

    df["lat_lon_interaction"] = (
        df["latitude"] * df["longitude"]
    )

    df["temp_lane_interaction"] = (
        df["Temperature"] * df["NumberofLanes"]
    )

    df["hour_lane_interaction"] = (
        df["hour"] * df["NumberofLanes"]
    )

    # --------------------------------------------------------
    # WEATHER ENCODING
    # --------------------------------------------------------

    weather_map = {
        "Sunny": 0,
        "Foggy": 1,
        "Rainy": 2,
        "Snowy": 3
    }

    df["weather_encoded"] = (
        df["Weather"].map(weather_map)
    )

    df["weather_encoded"] = (
        df["weather_encoded"].fillna(0)
    )

    df["weather_temp"] = (
        df["weather_encoded"] * df["Temperature"]
    )

    return df

# APPLY FEATURE ENGINEERING
train = create_features(train)
test = create_features(test)

# ============================================================
# FEATURES & TARGET
# ============================================================

X = train.drop(columns=[TARGET])
y = train[TARGET]

X_test = test.copy()

# ============================================================
# CATEGORICAL FEATURES
# ============================================================

cat_features = [
    "geohash",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather",
    "timestamp"
]

# CONVERT TO STRING
for col in cat_features:

    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# ============================================================
# CROSS VALIDATION
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof_preds = np.zeros(len(X))

test_preds_cat = np.zeros(len(X_test))
test_preds_lgb = np.zeros(len(X_test))

# ============================================================
# TRAINING LOOP
# ============================================================

for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):

    print("\n" + "="*50)
    print(f"FOLD {fold+1}")
    print("="*50)

    X_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx].copy()

    X_valid = X.iloc[valid_idx].copy()
    y_valid = y.iloc[valid_idx].copy()

    # ========================================================
    # CATBOOST MODEL
    # ========================================================

    cat_model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        loss_function='RMSE',
        eval_metric='R2',
        random_seed=42,
        verbose=200
    )

    cat_model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=200,
        use_best_model=True
    )

    valid_preds_cat = cat_model.predict(X_valid)

    test_fold_preds_cat = cat_model.predict(X_test)

    # ========================================================
    # LIGHTGBM PREP
    # ========================================================

    X_train_lgb = X_train.copy()
    X_valid_lgb = X_valid.copy()
    X_test_lgb = X_test.copy()

    for col in cat_features:

        combined = pd.concat([
            X_train_lgb[col],
            X_valid_lgb[col],
            X_test_lgb[col]
        ])

        unique_values = combined.unique()

        mapping = {
            k:v for v, k in enumerate(unique_values)
        }

        X_train_lgb[col] = (
            X_train_lgb[col].map(mapping)
        )

        X_valid_lgb[col] = (
            X_valid_lgb[col].map(mapping)
        )

        X_test_lgb[col] = (
            X_test_lgb[col].map(mapping)
        )

    # ========================================================
    # LIGHTGBM MODEL
    # ========================================================

    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=128,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    lgb_model.fit(
        X_train_lgb,
        y_train,
        eval_set=[(X_valid_lgb, y_valid)],
        eval_metric='l2'
    )

    valid_preds_lgb = lgb_model.predict(X_valid_lgb)

    test_fold_preds_lgb = lgb_model.predict(X_test_lgb)

    # ========================================================
    # ENSEMBLE
    # ========================================================

    valid_preds = (
        0.7 * valid_preds_cat +
        0.3 * valid_preds_lgb
    )

    # SAVE OOF
    oof_preds[valid_idx] = valid_preds

    # SAVE TEST PREDS
    test_preds_cat += (
        test_fold_preds_cat / kf.n_splits
    )

    test_preds_lgb += (
        test_fold_preds_lgb / kf.n_splits
    )

    # FOLD SCORE
    fold_r2 = r2_score(
        y_valid,
        valid_preds
    )

    print(f"\nFold R² Score: {fold_r2:.6f}")

# ============================================================
# FINAL CV SCORE
# ============================================================

final_r2 = r2_score(y, oof_preds)

print("\n" + "="*60)
print(f"FINAL CROSS-VALIDATED R² SCORE: {final_r2:.6f}")
print("="*60)

# ============================================================
# FINAL TEST PREDICTIONS
# ============================================================

final_test_preds = (
    0.7 * test_preds_cat +
    0.3 * test_preds_lgb
)

# REMOVE NEGATIVES
final_test_preds = np.clip(
    final_test_preds,
    0,
    None
)

# ============================================================
# CREATE SUBMISSION FILE
# ============================================================

submission = pd.DataFrame({
    "Index": test["Index"],
    "demand": final_test_preds
})

submission.to_csv(
    "/content/submission.csv",
    index=False
)

print("\nsubmission.csv created successfully!")

print("\nSUBMISSION PREVIEW:")
print(submission.head())

# ============================================================
# DONE
# ============================================================