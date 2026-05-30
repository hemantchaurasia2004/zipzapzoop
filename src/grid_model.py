"""Optional lightweight grid CNN for spatial demand (Colab GPU)."""
from __future__ import annotations

import json
from pathlib import Path

import geohash
import numpy as np
import pandas as pd

from .config import MODEL_DIR, RANDOM_SEED


def _decode(g: str) -> tuple[float, float]:
    try:
        return geohash.decode(g)
    except Exception:
        return np.nan, np.nan


def build_demand_grid(train_df: pd.DataFrame, day: int = 48) -> tuple[np.ndarray, dict]:
    """Build (lat x lon x slots) demand tensor for one day."""
    sub = train_df[train_df["day"] == day].copy()
    coords = {g: _decode(g) for g in sub["geohash"].unique()}
    lats = sorted({c[0] for c in coords.values() if not np.isnan(c[0])})
    lons = sorted({c[1] for c in coords.values() if not np.isnan(c[1])})
    lat_idx = {v: i for i, v in enumerate(lats)}
    lon_idx = {v: i for i, v in enumerate(lons)}
    grid = np.zeros((len(lats), len(lons), 96), dtype=np.float32)
    for _, row in sub.iterrows():
        lat, lon = coords[row["geohash"]]
        if np.isnan(lat):
            continue
        grid[lat_idx[lat], lon_idx[lon], int(row["slot_idx"])] = row["demand"]
    meta = {"lats": lats, "lons": lons, "lat_idx": lat_idx, "lon_idx": lon_idx, "day": day}
    return grid, meta


def grid_lookup(grid: np.ndarray, meta: dict, geohash_str: str, slot_idx: int) -> float:
    lat, lon = _decode(geohash_str)
    if np.isnan(lat) or lat not in meta["lat_idx"] or lon not in meta["lon_idx"]:
        return np.nan
    return float(grid[meta["lat_idx"][lat], meta["lon_idx"][lon], slot_idx])


def train_grid_baseline(train_df: pd.DataFrame, output_dir: Path | None = None) -> pd.DataFrame:
    """
    Simple baseline: predict test demand = day-48 same slot from grid.
    Useful as sanity check or low-weight ensemble component.
    """
    from .data_loader import load_test

    output_dir = Path(output_dir or MODEL_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid, meta = build_demand_grid(train_df, day=48)
    np.save(output_dir / "grid_d48.npy", grid)
    (output_dir / "grid_meta.json").write_text(json.dumps(meta, default=str))

    test_df = load_test()
    preds = []
    for _, row in test_df.iterrows():
        v = grid_lookup(grid, meta, row["geohash"], int(row["slot_idx"]))
        preds.append(v if not np.isnan(v) else train_df["demand"].mean())

    return pd.DataFrame({"Index": test_df["Index"], "demand": np.clip(preds, 1e-7, 1.0)})


def try_torch_cnn(train_df: pd.DataFrame, val_df: pd.DataFrame) -> float | None:
    """Train tiny Conv2D if torch is available; return validation R² or None."""
    try:
        import torch
        import torch.nn as nn
        from sklearn.metrics import r2_score
    except ImportError:
        return None

    torch.manual_seed(RANDOM_SEED)
    grid48, meta = build_demand_grid(train_df, day=48)
    grid49, _ = build_demand_grid(val_df, day=49)

    x = torch.tensor(grid48.transpose(2, 0, 1)[None], dtype=torch.float32)  # 1 x T x H x W
    y = torch.tensor(grid49.transpose(2, 0, 1)[None], dtype=torch.float32)

    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(96, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 96, 3, padding=1),
            )

        def forward(self, t):
            return self.net(t)

    model = TinyCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for _ in range(30):
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = model(x).numpy()[0].transpose(1, 2, 0)
    # Compare on overlapping geohashes in val
    actual, predicted = [], []
    for _, row in val_df.iterrows():
        lat, lon = _decode(row["geohash"])
        if lat not in meta["lat_idx"] or lon not in meta["lon_idx"]:
            continue
        i, j = meta["lat_idx"][lat], meta["lon_idx"][lon]
        s = int(row["slot_idx"])
        actual.append(row["demand"])
        predicted.append(pred[i, j, s])
    if len(actual) < 10:
        return None
    return float(r2_score(actual, predicted))
