"""Project paths and hyperparameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"
FEATURES_CACHE = OUTPUT_DIR / "features_cache.parquet"

RANDOM_SEED = 42
DAY_TRAIN_START = 48
DAY_HOLDOUT = 49
TEST_DAY = 49
TEST_TS_MIN = (2, 15)
TEST_TS_MAX = (13, 45)
SLOTS_PER_DAY = 96
SMOOTHING_M = 10

CAT_FEATURES = [
    "geohash",
    "RoadType",
    "Weather",
    "LargeVehicles",
    "Landmarks",
    "geohash_prefix2",
    "geohash_prefix4",
    "weather_hour",
    "road_lanes",
]

CATBOOST_PARAMS = {
    "loss_function": "RMSE",
    "iterations": 3000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 6,
    "random_seed": RANDOM_SEED,
    "verbose": 200,
    "early_stopping_rounds": 150,
}

LIGHTGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 3000,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": 8,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

BLEND_WEIGHT_CATBOOST = 0.55
OPTUNA_TRIALS = 40
