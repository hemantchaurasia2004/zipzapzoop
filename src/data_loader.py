"""Load and parse dataset CSVs."""
import pandas as pd

from .config import TRAIN_PATH, TEST_PATH


def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Parse H:M timestamp into hour, minute, slot_idx."""
    out = df.copy()
    parts = out["timestamp"].astype(str).str.split(":", expand=True)
    out["hour"] = parts[0].astype(int)
    out["minute"] = parts[1].astype(int)
    out["slot_idx"] = out["hour"] * 4 + (out["minute"] // 15)
    out["time_key"] = out["hour"] * 60 + out["minute"]
    return out


def timestamp_in_test_window(ts: str) -> bool:
    parts = str(ts).split(":")
    h, m = int(parts[0]), int(parts[1])
    t = h * 60 + m
    lo = 2 * 60 + 15
    hi = 13 * 60 + 45
    return lo <= t <= hi


def load_train() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_PATH)
    return parse_timestamp(df)


def load_test() -> pd.DataFrame:
    df = pd.read_csv(TEST_PATH)
    return parse_timestamp(df)


def get_holdout_mask(df: pd.DataFrame) -> pd.Series:
    """Day 49 rows present in train (early slots 0:00–2:00) for temporal validation."""
    return df["day"] == 49


def split_train_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on day 48; validate on day 49 rows in train (transfer to next day)."""
    train = df.loc[df["day"] == 48].copy()
    val = df.loc[df["day"] == 49].copy()
    if len(val) == 0:
        # Fallback: last 10% of day 48 by slot
        d48 = df.loc[df["day"] == 48].sort_values("slot_idx")
        cut = int(len(d48) * 0.9)
        train, val = d48.iloc[:cut].copy(), d48.iloc[cut:].copy()
    return train, val
